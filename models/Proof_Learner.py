import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base import BaseLearner
from utils.cluster import *
from utils.inc_net import Proof_Net
from utils.losses import *

num_workers = 8
class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args=args

        self._train_transformer=False
        self._network = Proof_Net(args, False)

        self.batch_size = get_attribute(args,"batch_size", 64)
        self.setting = get_attribute(args,"setting", "proof")
        self.train_templates = get_attribute(args,"train_templates", "one")
        self.text_optimize = get_attribute(args,"text_optimize", "loop")
        self.target_choose = get_attribute(args,"target_choose", "reselect")
        self.target_match = get_attribute(args, "target_match", "cosine")
        self.optimize_feat = get_attribute(args,"optimize_feat", "textimage")
        self.proto_select = get_attribute(args,"proto_select", "max")
        self.prediction = get_attribute(args,"prediction", "pmvmtm")
        self.lossteam = get_attribute(args,"lossteam", "pmvmtmncscmp")
        self.increment = get_attribute(args,"increment", 10)
        self.proto_num = get_attribute(args,"proto_num", 4)
        self.seed = get_attribute(args,"seed", 1993)
        self.num_classes = get_attribute(args, 'num_classes', 100)
        self.gen_proto_mode = get_attribute(args,"gen_proto_mode", "kmeans++")
        self.init_lr = get_attribute(args, "init_lr", 0.02)
        self.weight_decay = get_attribute(args, "weight_decay", 0.0005)
        self.min_lr = get_attribute(args, "min_lr", 1e-8)
        self.frozen_layers = get_attribute(args, "frozen_layers", None)

        self.tuned_epoch = get_attribute(args, "tuned_epoch", 5)
        self.ncloss = get_attribute(args, "ncloss", 10)
        self.scmploss = get_attribute(args, "scmploss", 3)

        self._known_classes = 0

        self.use_cos = get_attribute(args, "use_cos", False)

    def after_task(self):
        self._known_classes = self._total_classes
        print("Exemplar size: {}".format(self.exemplar_size))

    def cal_prototype(self,trainloader, model):
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_,data,label)=batch
                data=data.to(self._device)
                label=label.to(self._device)
                embedding=model.convnet.encode_image(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list=list(range(self._known_classes, self._total_classes))
        for class_index in class_list:
            data_index=(label_list==class_index).nonzero().squeeze(-1)
            embedding=embedding_list[data_index]
            if "mp" in self.setting:
                centroids = gen_mc_proto(embedding,self.proto_num,self.gen_proto_mode,self.seed)
                proto = centroids
            else:
                proto=embedding.mean(0)
            self._network.img_prototypes[class_index]=proto

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        self._network.update_prototype(self._total_classes)  # copy旧类别的 prototype，增加新 task 的 prototype
        self._network.update_context_prompt()  # add context prompts
        self._network.extend_projection()     # 增加新的 projection 层

        print("Learning on {}-{}".format(self._known_classes, self._total_classes))
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
            source="train", mode="train", appendent=self._get_memory())  #train data 里有旧 class 的数据
        self.train_dataset=train_dataset
        self.data_manager=data_manager
        self._network.to(self._device)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        # 取出当前 task 的 class 的数据，
        train_dataset_for_protonet=data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test" )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        #  先送入预训练 image encoder 计算 新 task 的 class prototype，不经过 projection 层
        self.cal_prototype(self.train_loader_for_protonet, self._network)
        acc,task_acc = self._train_proj(self.train_loader, self.test_loader, self.train_loader_for_protonet)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        return acc,task_acc

    def _train_proj(self, train_loader, test_loader, train_loader_for_protonet):
        self._train_transformer=True
        self._network.to(self._device)

        for name, param in self._network.convnet.named_parameters():
            if 'logit_scale' not in name:
                param.requires_grad = False
        self._network.freeze_projection_weight_new()

        if self.args['optimizer']=='sgd':
            optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr,weight_decay=self.weight_decay)
        elif self.args['optimizer']=='adam':
            optimizer = optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)

        class_to_label = self.data_manager._class_to_label
        if self.train_templates == "one":
            templates = self.data_manager._data_to_prompt[0]
        else:
            templates = self.data_manager._data_to_prompt
        prog_bar = tqdm(range(self.tuned_epoch))
        cliploss = ClipLoss()
        total_cls_names = class_to_label[:self._total_classes] # mask all known classes
        seen_class = list(range(self._total_classes))
        self.seen_class = seen_class
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                cls_names = [class_to_label[y] for y in targets]  #把 target 数字变成 class name
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                if self.train_templates == "one":
                    texts = [templates.format(inst) for inst in total_cls_names]
                    texts = self._network.tokenizer(texts).to(self._device)
                    text_features = self._network.encode_text(texts) # [total_classes, dim]
                    text_feas = text_features / text_features.norm(dim=-1, keepdim=True)
                else:
                    text_feas = self.use_templates_mean(total_cls_names,templates,self.text_optimize)

                image_features = self._network.encode_image(inputs)
                img_feas = image_features / image_features.norm(dim=-1, keepdim=True) #[bs, dim]

                prototype_features1 = self._network.encode_prototpyes(normalize=True)
                # if "mp" in self.setting:
                #     sepera_loss = separation_loss_cosine(prototype_features1)
                # prototype_features1 = compute_aug_proto(image_features, prototype_features1)
                use_multi_proto = "mp" in self.setting
                image_features, text_features, logit_scale, proto_features = self._network.forward_transformer(
                    img_feas, text_feas, self._train_transformer, prototype_features=prototype_features1,
                    use_multi_proto=use_multi_proto)
                if "nc" in self.setting:
                    # 把 text feature 往 ETF 上去拉，根据特征相似度来取对应的 target
                    loss_etf1 = torch.tensor(0.0, device=self._device)
                    loss_etf2 = torch.tensor(0.0, device=self._device)
                    if 'text' in self.optimize_feat:
                        loss_etf1 = self._network.eft_head.forward_train_v1(text_features, seen_class)["loss"]   #把 text feature 往随机初始化的 etf 上拉没有效果
                    if 'image' in self.optimize_feat:
                        if "mp" in self.setting:
                            # loss_etf2 = sum([self._network.eft_head.forward_train_v1(proto_features[:,po,:].squeeze(1), seen_class)["loss"] for po in range(self.proto_num)])
                            # 最优方案（兼顾效率和可维护性）
                            per_proto = proto_features.permute(1, 0, 2).contiguous()  # 必须保留.contiguous()
                            loss_etf2 = torch.stack([self._network.eft_head.forward_train_v1(per_proto[po], seen_class)["loss"] for po in range(self.proto_num)]).sum()
                            # loss_etf2 = self._network.eft_head.forward_train_v1(proto_features.mean(1), seen_class)["loss"]
                        else:
                            loss_etf2 = self._network.eft_head.forward_train_v1(proto_features, seen_class)["loss"]   #把 text feature 往随机初始化的 etf 上拉没有效果
                    # loss_etf3 = self._network.eft_head.forward_train_v1(image_features, [i.item() for i in targets])["loss"]   #把 text feature 往随机初始化的 etf 上拉没有效果
                    # loss_etf2+=loss_etf3
                    loss_etf = self.ncloss*(loss_etf1+loss_etf2)

                if "scmp" in self.setting: #supcontrast loss
                    # suploss = SupConLossMultiProto()  #这个效果不如SupConLoss， 尝试多prototype选择最好的，然后用下面的单prototype
                    # protoloss = suploss(image_features,proto_features,targets)
                    proto_features = get_similar_proto(image_features,proto_features,self.proto_select,targets)
                    suploss = Suploss_batchproto()
                    protoloss = self.scmploss * suploss(image_features, proto_features, targets)
                elif "sc" in self.setting: #supcontrast loss
                    suploss = SupConLoss()
                    protoloss = self.scmploss*suploss(image_features,proto_features,targets)
                elif "mp" in self.setting:
                    sepera_loss = separation_loss_cosine_2(proto_features)
                    protoloss = multiproto_max(image_features, proto_features, targets)
                    protoloss +=sepera_loss
                    # protoloss = F.cross_entropy(image_features @ proto_features[:, 0, :].squeeze(1).T, targets)
                    # protoloss += F.cross_entropy(image_features @ proto_features[:, 1, :].squeeze(1).T, targets)
                    # protoloss += F.cross_entropy(image_features @ proto_features[:, 2, :].squeeze(1).T, targets)
                    # protoloss += F.cross_entropy(image_features @ proto_features[:, 3, :].squeeze(1).T, targets)
                    # protoloss += sepera_loss
                else:
                    # proto_features = compute_aug_proto(image_features,proto_features)
                    # proto_features = compute_aug_proto(text_features,proto_features)
                    protoloss = F.cross_entropy(image_features @ proto_features.T, targets)

                logits = image_features@text_features.T # [bs, allclasses]

                if self.train_templates == "one":
                    texts=[templates.format(inst) for inst in cls_names]
                    clip_text_feas=self._network.encode_text(self._network.tokenizer(texts).to(self._device))#total,dim
                    clip_text_feas = clip_text_feas / clip_text_feas.norm(dim=-1, keepdim=True)
                else:
                    clip_text_feas = self.use_templates_mean(cls_names,templates,self.text_optimize)

                clip_loss = cliploss(img_feas, clip_text_feas, logit_scale)  # 这个loss有效

                loss = F.cross_entropy(logits, targets)

                total_loss = torch.tensor(0.0, device=self._device)
                if 'pm' in self.lossteam:
                    total_loss += clip_loss
                if 'tm' in self.lossteam:
                    total_loss += loss
                if 'nc' in self.lossteam:
                    total_loss += loss_etf
                if 'scmp' in self.lossteam:
                    total_loss += protoloss
                # if "nc" in self.setting:
                #     total_loss = loss+clip_loss+protoloss + loss_etf
                # else:
                #     total_loss = loss+clip_loss+protoloss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                losses += total_loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc ,task_acc= self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_acc {:.2f}, Test_acc {:.2f}".format(
                self._cur_task,epoch + 1,self.args['tuned_epoch'],losses / len(train_loader),train_acc, test_acc,  )
            # prog_bar.set_description(info)
            print(info)
        if "nc" in self.setting and self.target_choose == "reselect":
            self._network.eft_head.clear_assignment(self._total_classes)
        return test_acc,task_acc


    def use_templates_mean(self,total_cls_names,templates,text_optimize):
        if text_optimize == "loop":
            text_features = []
            for l in total_cls_names:
                texts = [t.format(l) for t in templates]
                texts = self._network.tokenizer(texts).to(self._device)
                class_embeddings = self._network.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                text_features.append(class_embeddings)
            text_feas = torch.stack(text_features, dim=0)
        else:
        # clip_text_features = []
        # for l in cls_names:
        #     texts = [t.format(l) for t in templates]
        #     texts = self._network.tokenizer(texts).to(self._device)
        #     class_embeddings = self._network.encode_text(texts)
        #     class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        #     class_embeddings = class_embeddings.mean(dim=0)
        #     class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        #     clip_text_features.append(class_embeddings)
        # clip_text_feas = torch.stack(clip_text_features, dim=0)

        #下面的代码是上面代码的高效改进版   但是内存占用会不断飙升
        # 批量生成所有类别和模板的组合文本
            texts = [t.format(cls_name) for cls_name in total_cls_names for t in templates]

            # 批量进行tokenize和文本编码（避免循环内重复调用）
            tokenized_texts = self._network.tokenizer(texts).to(self._device)
            all_embeddings = self._network.encode_text(tokenized_texts)  # [num_classes * num_templates, dim]

            # 对每个嵌入向量进行归一化（合并两次归一化为一次）
            all_embeddings = torch.nn.functional.normalize(all_embeddings, dim=-1)  # [num_classes * num_templates, dim]

            # 重组为三维张量并计算均值
            text_feas = all_embeddings.view(
                len(total_cls_names),
                len(templates),
                -1
            ).mean(dim=1)  # [num_classes, dim]

            # 对最终结果进行整体归一化
            text_feas = torch.nn.functional.normalize(text_feas, dim=-1)

        return text_feas


    def _compute_accuracy(self, model, loader):# 只计算 top1
        self._network.eval()
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt
        total_labels = class_to_label[:self._total_classes] # mask all known classes
        text_features = []
        with torch.no_grad():
            # for l in total_labels:
            #     texts = [t.format(l) for t in templates]
            #     texts = self._network.tokenizer(texts).to(self._device)
            #     class_embeddings = self._network.encode_text(texts)
            #     class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            #     class_embeddings = class_embeddings.mean(dim=0)
            #     class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            #     text_features.append(class_embeddings)
            # text_features = torch.stack(text_features, dim=0)
            text_features = self.use_templates_mean(total_labels,templates,self.text_optimize)  #这里本来就是用所有的  ,这个写法跟上面的代码效果完全一样，但是内存占用会不断飙升

        # use_multi_proto = True
        use_multi_proto = "mp" in self.setting
        # if not use_multi_proto:     #这个是给有多个prototype但是test时只用一个的setting使用的
        #     self._network.img_prototypes_co = deepcopy(self._network.img_prototypes)
        #     self._network.img_prototypes = self._network.img_prototypes[:,0,:].squeeze(1)
        correct, total = 0, 0
        task_correct = {}
        task_total = {}
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                image_features=self._network.encode_image(inputs)
                transf_image_features, transf_text_features, _, proto_feas = self._network.forward_transformer(
                    image_features, text_features,self._train_transformer,use_multi_proto=use_multi_proto)
                if "mp" in self.setting or use_multi_proto:
                    img_expanded = transf_image_features.view(transf_image_features.shape[0], 1, 1, proto_feas.shape[2])  # (B, 1, 1, D)
                    proto_expanded = proto_feas.view(1, proto_feas.shape[0], proto_feas.shape[1], proto_feas.shape[2])  # (1, C, P, D)
                    # 计算余弦相似度：A_expanded 和 B 的形状是 (64, 3, 512)，我们可以计算它们的点积
                    cos_sim = F.cosine_similarity(img_expanded, proto_expanded, dim=-1)  # 输出的形状是 (64, 3)
                    # 从每个样本中选择最大相似度的索引
                    # max_sim_values 是相似度值，可以作为 logits，max_sim_indices 是选择的相似向量在 3 维度中的索引
                    max_sim_values, max_sim_indices = cos_sim.max(dim=2)
                    proto_feas = proto_feas[range(proto_feas.shape[0]), max_sim_indices]  # 选择每个样本最相似的向量 64，10，,512

                    proto_outputs = (transf_image_features.unsqueeze(1) * proto_feas).sum(-1)

                else:
                    proto_outputs= transf_image_features @ proto_feas.T

                tmoutputs = transf_image_features @ transf_text_features.T
                original_outputs= image_features @ text_features.T
                # outputs = original_outputs+tmoutputs+proto_outputs

                outputs = torch.zeros_like(proto_outputs)
                if 'pm' in self.prediction:
                    outputs += original_outputs
                if 'vm' in self.prediction:
                    outputs += proto_outputs
                if 'tm' in self.prediction:
                    outputs += tmoutputs

            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

            tasks = targets // self.increment
            # 遍历当前 batch 中所有 task（去重）
            for task in tasks.unique():
                # 构造 mask，筛选出属于当前 task 的样本
                mask = (tasks == task)
                # 累计该 task 中的正确预测数
                correct_t = (predicts.cpu()[mask] == targets[mask]).sum().item()
                total_t = mask.sum().item()

                taskNo = int(task.item())  # 转为 Python int
                task_correct[taskNo] = task_correct.get(taskNo, 0) + correct_t
                task_total[taskNo] = task_total.get(taskNo, 0) + total_t

        # 计算每个 task 的准确率，按照 task 编号升序排列
        task_accuracies = []
        for task in sorted(task_total.keys()):
            acc = task_correct[task] / task_total[task]
            task_accuracies.append([acc, task_correct[task] , task_total[task]])
        # task_accuracies = [np.around(accru*100, decimals=2) for accru in task_accuracies]
        # if not use_multi_proto:
        #     self._network.img_prototypes = self._network.img_prototypes_co
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2),task_accuracies


    def get_last_proto(self):# 只计算 top1
        self._network.eval()
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt
        total_labels = class_to_label # mask all known classes
        with torch.no_grad():
            text_features = self.use_templates_mean(total_labels,templates,self.text_optimize)  #这里本来就是用所有的  ,这个写法跟上面的代码效果完全一样，但是内存占用会不断飙升

        use_multi_proto = "mp" in self.setting
        final_protos=[]
        final_images=[]
        final_texts=[]
        for class_idx in range(self._total_classes):
            class_dset = self.data_manager.get_dataset(np.arange(class_idx, class_idx + 1), source="test", mode="test")
            class_loader = DataLoader(class_dset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
            for i, (_, inputs, targets) in enumerate(class_loader):
                inputs = inputs.to(self._device)
                with torch.no_grad():
                    image_features=self._network.encode_image(inputs)     #todo 要把test feature加进去，那就是两张图对比，fusion之后的image feature和prototype
                    transf_image_features, transf_text_features, _, proto_feas = self._network.forward_transformer(
                        image_features, text_features,self._train_transformer,use_multi_proto=use_multi_proto)
                    final_protos.append(proto_feas.view(self._total_classes,-1,self.feature_dim)[class_idx,:,:])
                    final_images.append(transf_image_features)
                    final_texts.append(transf_text_features)
                break
        final_protos = torch.cat(final_protos,dim=0)
        final_images = torch.stack(final_images,dim=0)
        final_texts = torch.stack(final_texts,dim=0)
        save_tensors(self.setting,final_protos)
        save_tensors(f"{self.setting}_class_image",final_images)
        save_tensors(f"{self.setting}_class_text",final_texts)
        if "nc" in self.setting:
            save_tensors(f"{self.setting}_etf",self._network.eft_head.etf_vec)
        print(f"{final_protos.shape},{final_images.shape},{final_texts.shape}")
        print("save proto finish")

    def _eval_cnn(self, loader):#会计算topK

        self._network.eval()
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt
        total_labels = class_to_label[:self._total_classes] # mask all known classes
        text_features = []
        with torch.no_grad():
            for l in total_labels:
                texts = [t.format(l) for t in templates]
                texts = self._network.tokenizer(texts).to(self._device)
                class_embeddings = self._network.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=0)

        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                image_features=self._network.encode_image(inputs)
                transf_image_features, transf_text_features, _, proto_feas = self._network.forward_transformer(image_features, text_features,self._train_transformer)

                outputs = transf_image_features @ transf_text_features.T

                proto_outputs= transf_image_features @ proto_feas.T

                original_outputs= image_features @ text_features.T

                outputs = original_outputs+outputs+proto_outputs

            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]