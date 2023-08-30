import copy
import math
import torch
import warnings
import time
import numpy as np
from torch import nn
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.nn import Module, Parameter
from torch.utils.data import DataLoader
from .LAS_utils import mixup_data, mixup_criterion,LabelAwareSmoothing, LearnableWeightScaling
import  datasets.data_loader as stage2_utils
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from itertools import product


class IntermediateClassSampler():

    def __init__(self, label, pairs ):
        #self.n_batch = n_batch  # the number of iterations in the dataloader
        #self.n_cls = n_cls
        #self.n_per = n_per
        self.pairs = pairs
        self.label = np.array(label)  # all data label
        self.m_ind = []  # the data index of each class
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

        self.int_classes_len = len(pairs)

    def __len__(self):
        return 1

    def __iter__(self):

        
        batch1 = []
        batch2 = []

        

        for int_class in range(len(self.label), len(self.label)+self.int_classes_len):
            cls_idx1, cls_idx2 = self.pairs[int_class]
            l1 = self.m_ind[cls_idx1]  # all data indexs of this class
            pos1 = torch.randperm(len(l1))[:1]  # sample n_per data index of this class
            batch1.append(l1[pos1])

            
            l2 = self.m_ind[cls_idx2]  # all data indexs of this class
            pos2 = torch.randperm(len(l2))[:1]  # sample n_per data index of this class
            batch2.append(l2[pos2])

        
        batch1 = torch.stack(batch1).reshape(-1)
        batch2 = torch.stack(batch2).reshape(-1)
        batch = torch.cat([batch1, batch2])
        # .t() transpose,
        # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
        # instead of aaaabbbbccccdddd
        yield batch

class Appr(Inc_Learning_Appr):
    """Class implementing the Learning a Unified Classifier Incrementally via Rebalancing (LUCI) approach
    described in http://dahua.me/publications/dhl19_increclass.pdf
    Original code available at https://github.com/hshustc/CVPR19_Incremental_Learning
    """

    # Sec. 4.1: "we used the method proposed in [29] based on herd selection" and "first one stores a constant number of
    # samples for each old class (e.g. R_per=20) (...) we adopt the first strategy"
    def __init__(self, model, device, nepochs=160, lr=0.1, lr_min=1e-4, lr_factor=10, lr_patience=8, clipgrad=10000,
                 momentum=0.9, wd=5e-4, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, logger=None, exemplars_dataset=None, lamb=10., lamb_mr=1., dist=0.5, K=2,
                 remove_less_forget=False, remove_margin_ranking=False, remove_adapt_lamda=False,schedule_step = [80,120]):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)                
        self.lamb = lamb
        self.lamb_mr = lamb_mr
        self.dist = dist
        self.K = K
        self.less_forget = not remove_less_forget
        self.margin_ranking = not remove_margin_ranking
        self.adapt_lamda = not remove_adapt_lamda
        self.lws_models = torch.nn.ModuleList()

        self.lamda = self.lamb
        self.ref_model = None

        self.warmup_loss = self.warmup_luci_loss
        # LUCIR is expected to be used with exemplars. If needed to be used without exemplars, overwrite here the
        # `_get_optimizer` function with the one in LwF and update the criterion
        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        if not have_exemplars:
            warnings.warn("Warning: LUCIR is expected to use exemplars. Check documentation.")

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Sec. 4.1: "lambda base is set to 5 for CIFAR100 and 10 for ImageNet"
        parser.add_argument('--lamb', default=5., type=float, required=False,
                            help='Trade-off for distillation loss (default=%(default)s)')
        # Loss weight for the Inter-Class separation loss constraint, set to 1 in the original code
        parser.add_argument('--lamb-mr', default=1., type=float, required=False,
                            help='Trade-off for the MR loss (default=%(default)s)')
        # Sec 4.1: "m is set to 0.5 for all experiments"
        parser.add_argument('--dist', default=.5, type=float, required=False,
                            help='Margin threshold for the MR loss (default=%(default)s)')
        # Sec 4.1: "K is set to 2"
        parser.add_argument('--K', default=2, type=int, required=False,
                            help='Number of "new class embeddings chosen as hard negatives '
                                 'for MR loss (default=%(default)s)')
        # Flags for ablating the approach
        parser.add_argument('--remove-less-forget', action='store_true', required=False,
                            help='Deactivate Less-Forget loss constraint(default=%(default)s)')
        parser.add_argument('--remove-margin-ranking', action='store_true', required=False,
                            help='Deactivate Inter-Class separation loss constraint (default=%(default)s)')
        parser.add_argument('--remove-adapt-lamda', action='store_true', required=False,
                            help='Deactivate adapting lambda according to the number of classes (default=%(default)s)')
        return parser.parse_known_args(args)
    def _get_optimizer(self):
        """Returns the optimizer"""
        if self.less_forget:
            # Don't update heads when Less-Forgetting constraint is activated (from original code)
            params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params = self.model.parameters()
        # for name,_ in self.model.named_parameters():
            # print(name)
        # breakpoint()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def protosave(self, model, loader, current_task, num_of_classes):
        print("saving protos...")
        features = []
        labels = []
        model.eval()
        with torch.no_grad():
            for images, targets in loader:
                images, target = images.to(self.device), targets.to(self.device)
            # Forward current model
                _, feature = model(images, return_features=True)
            # for i, (indexs, images, target) in enumerate(loader):
                # feature = model.feature(images.to(self.device))
                # if feature.shape[0] == self.args.batch_size:
                # labels.append(target.cpu().numpy())   
                # features.append(feature.cpu().numpy())
                
                # labels.append(target.view(-1,1))   
                # features.append(feature)

                labels.append(target.cpu())
                features.append(feature.cpu())


        # breakpoint()
        # labels = torch.vstack(labels)
        # labels = labels.squeeze(1)
        # features = torch.vstack(features)

        # print(labels.shape, features.shape)

        # print(c)

        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)
        # print(labels.shape, features.shape)
        # print(c)
        
        # labels = np.array(labels.cpu())
        # labels_set = np.unique(labels)
        # labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
        # features = np.array(features.cpu())
        # features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))
        # feature_dim = features.shape[1]

        # breakpoint()

        prototype = []
        radius = []
        class_label = []
        cov_list = []
        num_of_samples = []
        for class_index in range(num_of_classes):
            
            # breakpoint()
            data_index = (labels == class_index).nonzero()
            # print('class index', class_index, data_index.shape)
            embedding = features[data_index.squeeze(-1)]
            embedding = F.normalize(embedding, p=2, dim=-1)
            #print('dim of emd', embedding_this.shape)
            #print(c)
            feature_class_wise = embedding.numpy()
            cov = np.cov(feature_class_wise.T)
            cov_torch = torch.cov(embedding.T)
            # print('class index', class_index)
            # print('class index', class_index, torch.linalg.eig(cov_torch)[0])
            radius.append(np.trace(cov)/64)
            print('class index', class_index, 'number of samples',data_index.shape[0],'radius', radius[-1])
            num_of_samples.append(data_index.shape[0])
            embedding_mean = embedding.mean(0)
            prototype.append(embedding_mean )
            # cov_list.append(torch.tensor(cov))
            cov_list.append(cov_torch)
        
        self.radius = np.sqrt(np.mean(radius)) 
        self.radius_list = radius
        self.proto_list = torch.stack(prototype, dim=0)
        self.cov_list = torch.stack(cov_list, dim=0)
        self.num_of_samples = torch.tensor(num_of_samples)

    def pre_train_process(self, t, trn_loader):
        """Runs before training all epochs of the task (before the train session)"""
        lws_model = LearnableWeightScaling(num_classes=self.model.task_cls[t]).to(self.device)
        self.lws_models.append(lws_model)
        if t == 0:
            # Sec. 4.1: "the ReLU in the penultimate layer is removed to allow the features to take both positive and
            # negative values"
            if self.model.model.__class__.__name__ == 'ResNet':
                old_block = self.model.model.layer3[-1]
                self.model.model.layer3[-1] = BasicBlockNoRelu(old_block.conv1, old_block.bn1, old_block.relu,
                                                               old_block.conv2, old_block.bn2, old_block.downsample)
            else:
                warnings.warn("Warning: ReLU not removed from last block.")
        # Changes the new head to a CosineLinear
        self.model.heads[-1] = CosineLinear(self.model.heads[-1].in_features, self.model.heads[-1].out_features)
        
        ###########################################
        # breakpoint()
        # self.total_class = 100
        # self.augnumclass = self.total_class + int(self.total_class*(self.total_class-1)/2)
        # self.model.model.fc_int = CosineLinear(self.model.heads[-1].in_features, self.augnumclass)
        # breakpoint()
        ############################################
        self.model.to(self.device)
        if t > 0:
            # Share sigma (Eta in paper) between all the heads
            self.model.heads[-1].sigma = self.model.heads[-2].sigma
            # Fix previous heads when Less-Forgetting constraint is activated (from original code)
            if self.less_forget:
                for h in self.model.heads[:-1]:
                    for param in h.parameters():
                        param.requires_grad = False
                self.model.heads[-1].sigma.requires_grad = True
            # Eq. 7: Adaptive lambda
            if self.adapt_lamda:
                self.lamda = self.lamb * math.sqrt(sum([h.out_features for h in self.model.heads[:-1]])
                                                   / self.model.heads[-1].out_features)
        # The original code has an option called "imprint weights" that seems to initialize the new head.
        # However, this is not mentioned in the paper and doesn't seem to make a significant difference.
        super().pre_train_process(t, trn_loader)

    def train_loop(self, t, trn_loader, val_loader, args, tst_loader):
        """Contains the epochs loop"""

        
        ''' Stage1 normal training   '''
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                    batch_size=trn_loader.batch_size,
                                                    shuffle=True,
                                                    num_workers=trn_loader.num_workers,
                                                    pin_memory=trn_loader.pin_memory)


        for parameter in self.model.model.parameters():
            parameter.requires_grad = True
        print("trainable_parameters_list....")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)

        lr = self.lr
        best_loss = np.inf
        patience = self.lr_patience
        self.optimizer = self._get_optimizer()

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()
            self.train_epoch(t, trn_loader)
            clock1 = time.time()
            self.eval_on_train = False
            if self.eval_on_train:
                train_loss, train_acc, _ = self.eval(t, trn_loader)
                clock2 = time.time()
                print('| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=train_loss, group="train")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * train_acc, group="train")
            else:
                print('| Epoch {:3d}, time={:5.1f}s | Train: skip eval |'.format(e + 1, clock1 - clock0), end='')

            # Valid
            clock3 = time.time()
            valid_loss, valid_acc, _ = self.eval(t, val_loader)
            clock4 = time.time()
            print(' Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}% |'.format(
                clock4 - clock3, valid_loss, 100 * valid_acc), end='')
            self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=valid_loss, group="valid")
            self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * valid_acc, group="valid")

            # Adapt learning rate - patience scheme - early stopping regularization
            if valid_loss < best_loss:
                # if the loss goes down, keep it as the best model and end line with a star ( * )
                best_loss = valid_loss
                patience = self.lr_patience
                print(' *', end='')
            if e+1 in self.model.schedule_step:
                lr/=self.lr_factor
                print(' lr={:.1e}'.format(lr), end='')
                self.optimizer.param_groups[0]['lr'] = lr
                # self.model.set_state_dict(best_model)
            self.logger.log_scalar(task=t, iter=e + 1, name="patience", value=patience, group="train")
            self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=lr, group="train")
            print()
        #################################################################################
       
        # torch.save(self.model.state_dict(), '/home/teja/long_tail_cil/Long-Tailed-CIL/cifar100/50base_6tasks/results/test_models_for_adapt_before_test/task'+str(t)+'.ckpt')
        ''' Stage1 normal intemediate node training  '''
        for parameter in self.model.model.parameters():
            parameter.requires_grad = True
        print("trainable_parameters_list....")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)
                
                
        # breakpoint()
        ''' Intermediate nodes adding '''
        Current_weights = self.model.heads[0].weight.clone().detach()
        cls_sim = F.cosine_similarity(Current_weights.unsqueeze(1), Current_weights.unsqueeze(0), dim=-1, eps=1e-30)
        # breakpoint()
        sim_classes = {}
        dissim_classes = {}
        temp_pairs = {}
        n_way=Current_weights.shape[0]
        k = n_way
        for c in range(n_way):
            sorted, indices = torch.sort(-cls_sim[c])
            indices = indices.cpu().data.numpy()
            sim_classes[c] = indices[1:4]
            #dissim_classes[c] = indices[3:]

            pair_1, pair_2, pair_3 = list(product([c], sim_classes[c]))
            # pair_1, pair_2, pair_3, pair_4, pair_5,pair_6,pair_7,pair_8,pair_9,pair_10 = list(product([c], sim_classes[c]))
            # print(pair_c)
            temp_pairs[pair_1] = k
            temp_pairs[pair_2] = k+1
            temp_pairs[pair_3] = k+2
            k = k+3
            
            # temp_pairs[pair_4] = k+4
            # temp_pairs[pair_5] = k+5
            # temp_pairs[pair_6] = k+6

            # temp_pairs[pair_7] = k+7
            # temp_pairs[pair_8] = k+8
            # temp_pairs[pair_9] = k+9

            # temp_pairs[pair_10] = k+10
            
            
            # k = k+11
        # breakpoint()
            
        #print(c)
        # print('sim', sim_classes)
        # print('temp pairs', temp_pairs)

        pairs = {}
        for key, val in temp_pairs.items():
            pairs[key] = val
            pairs[(key[1], key[0])] = val
    # print('PAIRS')
    # print(pairs)

        final_dict ={} # create empty list
        for key, val in pairs.items(): 
            if val in final_dict.values(): 
                continue 
            else:
                final_dict[key] = val

    # print('final pairs')
    # print(final_dict)

        final_pairs = {}
        k = n_way
        for key, val in final_dict.items():
            final_pairs[k] = key
            k+=1

        print('final pairs', final_pairs)
        print('final pairs', len(final_pairs))
                # args.pairs = final_pairs
                
        self.model.model.fc_int = CosineLinear(self.model.heads[-1].in_features, len(final_pairs))
        # breakpoint()
        for cls_id in final_pairs.keys():
            C1, C2 = final_pairs[cls_id]
            self.model.model.fc_int.weight.data[cls_id - Current_weights.shape[0]] = (Current_weights[C1] + Current_weights[C2])/2
            
            # print(C1,C2) 
        self.model.to(self.device)
        # trainloader_class_aug = self.get_class_aug_mix_dataloader(final_pairs)
        
        # breakpoint()
        if len(self.exemplars_dataset) > 0 and t > 0:
            total_unique_targets = trn_loader.datset.labels + self.exemplars_dataset.labels
            
        else:
            total_unique_targets = trn_loader.dataset.labels 
        total_unique_targets = np.unique(total_unique_targets)
        # breakpoint()
        
        sampler = IntermediateClassSampler(total_unique_targets, final_pairs)
        
        if len(self.exemplars_dataset) > 0 and t > 0:

            trainloader_class_aug = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                                batch_sampler=sampler, 
                                                        # batch_size=trn_loader.batch_size,
                                                        # shuffle=True,
                                                        num_workers=trn_loader.num_workers,
                                                        pin_memory=trn_loader.pin_memory)
        else:
            

            trainloader_class_aug = torch.utils.data.DataLoader(trn_loader.dataset,
                                                                batch_sampler=sampler, 
                                                        # batch_size=trn_loader.batch_size,
                                                        # shuffle=True,
                                                        num_workers=trn_loader.num_workers,
                                                        pin_memory=trn_loader.pin_memory)
        
        
        label_list=[]
        for j in range(len(total_unique_targets), len(final_pairs)+len(total_unique_targets)):
            label_list.append(j)        
        # breakpoint()
        #################################################################################

        # lr = self.lr
        self.lr = self.lr * 0.1
        best_loss = np.inf
        patience = self.lr_patience
        self.optimizer = self._get_optimizer()

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()
            self.train_epoch_aug(t, trn_loader, trainloader_class_aug, label_list, final_pairs)
            clock1 = time.time()
            self.eval_on_train = False
            if self.eval_on_train:
                train_loss, train_acc, _ = self.eval(t, trn_loader)
                clock2 = time.time()
                print('| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=train_loss, group="train")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * train_acc, group="train")
            else:
                print('| Epoch {:3d}, time={:5.1f}s | Train: skip eval |'.format(e + 1, clock1 - clock0), end='')

            # Valid
            clock3 = time.time()
            valid_loss, valid_acc, _ = self.eval(t, val_loader)
            clock4 = time.time()
            print(' Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}% |'.format(
                clock4 - clock3, valid_loss, 100 * valid_acc), end='')
            self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=valid_loss, group="valid")
            self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * valid_acc, group="valid")

            # Adapt learning rate - patience scheme - early stopping regularization
            if valid_loss < best_loss:
                # if the loss goes down, keep it as the best model and end line with a star ( * )
                best_loss = valid_loss
                patience = self.lr_patience
                print(' *', end='')
            if e+1 in self.model.schedule_step:
                lr/=self.lr_factor
                print(' lr={:.1e}'.format(lr), end='')
                self.optimizer.param_groups[0]['lr'] = lr
                # self.model.set_state_dict(best_model)
            self.logger.log_scalar(task=t, iter=e + 1, name="patience", value=patience, group="train")
            self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=lr, group="train")
            print()
       #################################################################################
        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)
        #################################################################################
        
        ''' Task wise evaluation '''
        max_task = args.max_task
        acc_taw = np.zeros((max_task, max_task))
        acc_tag = np.zeros((max_task, max_task))
        forg_taw = np.zeros((max_task, max_task))
        forg_tag = np.zeros((max_task, max_task))

        for u in range(t + 1):
            # test_loss, acc_taw[t, u], acc_tag[t, u] = self.eval(u, tst_loader[u])
            test_loss, acc_taw[t, u], acc_tag[t, u], pred, targets = self.eval(u, tst_loader[u], return_pred_targets=True)
            
            if u < t:
                forg_taw[t, u] = acc_taw[:t, u].max(0) - acc_taw[t, u]
                forg_tag[t, u] = acc_tag[:t, u].max(0) - acc_tag[t, u]
            print('>>> Test on task {:2d} : loss={:.3f} | TAw acc={:5.1f}%, forg={:5.1f}%'
                  '| TAg acc={:5.1f}%, forg={:5.1f}% <<<'.format(u, test_loss,
                                                                 100 * acc_taw[t, u], 100 * forg_taw[t, u],
                                                                 100 * acc_tag[t, u], 100 * forg_tag[t, u]))

        #################################################################################
        self.lr = self.lr * 10
        
        ''' Global varaince tuning --->  Stage2'''
        unique_classes = set()

        for images, labels in trn_loader:
            unique_classes.update(labels.unique().tolist())

        num_unique_classes = len(unique_classes)

        print("Number of unique classes:", num_unique_classes)
        self.protosave(self.model, trn_loader,t, num_unique_classes)
        # print('checking..', self.proto_list.shape, self.cov_list.shape,self.radius)
        class_id_most_samples = torch.argmax(self.num_of_samples)
        if t==0:
            self.cov_cls_ms_major = self.cov_list[class_id_most_samples]

            # breakpoint()
        if t >= 0:

            cov_cls_ms = self.cov_cls_ms_major.repeat(num_unique_classes, 1, 1)
            mean = self.proto_list
            distrib = MultivariateNormal(loc=mean, covariance_matrix=cov_cls_ms)

            for parameter in self.model.model.parameters():
                parameter.requires_grad = False

            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

            for parameter in self.model.heads.parameters():
                parameter.requires_grad = True
            
            print("trainable_parameters_list....")
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(name)
            params = self.model.heads.parameters()
            lr = 0.1
            self.optimizer_classifier_tune = torch.optim.SGD(params, lr=lr, weight_decay=self.wd, momentum=self.momentum)

            # valid_loss, valid_acc, _ = self.eval_with_exemplars(t, val_loader)
            # print('Valid loss:', valid_loss, 'Acc', valid_acc)+
            for e in range(100):
                # Train
                clock0 = time.time()
                # self.train_epoch(t, trn_loader)
                self.optimizer_classifier_tune = self.classifier_tune(t, distrib, num_unique_classes)
                clock1 = time.time()
                self.eval_on_train = False
                if self.eval_on_train:
                    train_loss, train_acc, _ = self.eval(t, trn_loader)
                    clock2 = time.time()
                    print('| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                        e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')
                    self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=train_loss, group="train")
                    self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * train_acc, group="train")
                else:
                    print('| Epoch {:3d}, time={:5.1f}s | Train: skip eval |'.format(e + 1, clock1 - clock0), end='')

                # Valid
                clock3 = time.time()
                valid_loss, valid_acc, _ = self.eval(t, val_loader)
                clock4 = time.time()
                print(' Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    clock4 - clock3, valid_loss, 100 * valid_acc), end='')
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=valid_loss, group="valid")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * valid_acc, group="valid")

                # Adapt learning rate - patience scheme - early stopping regularization
                if valid_loss < best_loss:
                    # if the loss goes down, keep it as the best model and end line with a star ( * )
                    best_loss = valid_loss
                    patience = self.lr_patience
                    print(' *', end='')
                if e+1 in self.model.schedule_step:
                    lr/=self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    self.optimizer_classifier_tune.param_groups[0]['lr'] = lr
                    # self.model.set_state_dict(best_model)
                self.logger.log_scalar(task=t, iter=e + 1, name="patience", value=patience, group="train")
                self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=lr, group="train")
                print()

            valid_loss, valid_acc, _ = self.eval(t, val_loader)
            print('Valid loss:', valid_loss, 'Acc', valid_acc)

            #################################################################################
            ''' Task wise evaluation '''
            max_task = args.max_task
            acc_taw = np.zeros((max_task, max_task))
            acc_tag = np.zeros((max_task, max_task))
            forg_taw = np.zeros((max_task, max_task))
            forg_tag = np.zeros((max_task, max_task))

            for u in range(t + 1):
                # test_loss, acc_taw[t, u], acc_tag[t, u] = self.eval(u, tst_loader[u])
                test_loss, acc_taw[t, u], acc_tag[t, u], pred, targets = self.eval(u, tst_loader[u], return_pred_targets=True)
            
                if u < t:
                    forg_taw[t, u] = acc_taw[:t, u].max(0) - acc_taw[t, u]
                    forg_tag[t, u] = acc_tag[:t, u].max(0) - acc_tag[t, u]
                print('>>> Test on task {:2d} : loss={:.3f} | TAw acc={:5.1f}%, forg={:5.1f}%'
                    '| TAg acc={:5.1f}%, forg={:5.1f}% <<<'.format(u, test_loss,
                                                                    100 * acc_taw[t, u], 100 * forg_taw[t, u],
                                                                    100 * acc_tag[t, u], 100 * forg_tag[t, u]))

         #################################################################################

  
    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval()
        # Make the old model return outputs without the sigma (eta in paper) factor
        for h in self.ref_model.heads:
            h.train()
        self.ref_model.freeze_all()
    
    def mixup_data(self, x, y, alpha=1.0, use_cuda=True):
        # print(alpha)
        
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam, 0

    def mixup_criterion(self, pred, y_a, y_b, lam, org_bs):
        pred = torch.cat([o['wsigma'] for o in pred], dim=1)[org_bs:]
        return lam * nn.CrossEntropyLoss()(pred, y_a) + (1 - lam) * nn.CrossEntropyLoss()(pred, y_b)
    
    # def mixup_data_discrete(self, x, y, alpha=1.0, use_cuda=True):
    #     '''Returns mixed inputs, pairs of targets, and lambda'''
    #     # if alpha > 0:
    #     #     lam = np.random.beta(alpha, alpha)
    #     # else:
    #     #     lam = 1

    #     batch_size = x.size()[0]
    #     sample_list = [0.25, 0.5, 0.75]
    #     tensor_shape = [batch_size,1]
    #     random_indices = torch.randint(low=0, high=len(sample_list), size=tensor_shape)
    #     lam = torch.Tensor([sample_list[i] for i in random_indices])
    #     lam = lam.reshape(-1, 1).cuda()
    #     if use_cuda:
    #         index = torch.randperm(batch_size).cuda()
    #     else:
    #         index = torch.randperm(batch_size)
    #     # breakpoint()
    #     # mixed_x = lam * x + (1 - lam) * x[index, :]
    #     mixed_x = x * lam[:, None, None] + x[index, :] * (1 - lam[:, None, None])
    #     y_a, y_b = y, y[index]
    #     # breakpoint()
    #     return mixed_x, y_a, y_b, lam, index


    # def mixup_criterion_discrete(self,pred, y_a, y_b, lam, org_bs):
    #     pred = torch.cat([o['wsigma'] for o in pred], dim=1)[org_bs:]
    #     criterion_mix = nn.CrossEntropyLoss(reduction='none')
    #     # breakpoint()
    #     return torch.mean(lam.reshape(-1) * criterion_mix(pred, y_a) + (1 - lam).reshape(-1) * criterion_mix(pred, y_b))
    def classifier_tune(self, t, distrib, num_of_classes):
        """Runs a single epoch"""
        # self.lws_models.eval()
        self.model.train()
        # for parameter in self.model.model.parameters():
        #     parameter.requires_grad = False

        # for m in self.model.modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.eval()

        # for parameter in self.model.heads.parameters():
        #     parameter.requires_grad = True
        
        # print("trainable_parameters_list....")
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(name)
        # params = self.model.heads.parameters()
        # lr = 0.1
        # self.optimizer_classifier_tune = torch.optim.SGD(params, lr=lr, weight_decay=self.wd, momentum=self.momentum)

        samples = distrib.rsample()
        # breakpoint()
        targets = torch.arange(num_of_classes).cuda()
        # print(sample.shape, targets)
        outputs = []
        for head in self.model.heads:
            outputs.append(head(samples.cuda().float()))
        outputs = torch.cat([o['wsigma'] for o in outputs], dim=1)

        # outputs = self.model.heads[-1](samples.cuda().float())
        # breakpoint()

        loss = nn.CrossEntropyLoss(None)(outputs, targets.long())
        # Backward
        self.optimizer_classifier_tune.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
        self.optimizer_classifier_tune.step()

        return self.optimizer_classifier_tune
    
    def classAug(self,x, y,alpha=20.0, mix_times=4):  # mixup based
        batch_size = x.size()[0]
        mix_data = []
        mix_target = []
        for _ in range(mix_times):
            index = torch.randperm(batch_size).cuda()
            for i in range(batch_size):
                if y[i] != y[index][i]:
                    new_label = self.generate_label(y[i].item(), y[index][i].item())
                    lam = np.random.beta(alpha, alpha)
                    if lam < 0.4 or lam > 0.6:
                        lam = 0.5
                    mix_data.append(lam * x[i] + (1 - lam) * x[index, :][i])
                    mix_target.append(new_label)

        new_target = torch.Tensor(mix_target)
        y = torch.cat((y, new_target.cuda().long()), 0)
        for item in mix_data:
            x = torch.cat((x, item.unsqueeze(0)), 0)
        return x, y

    def generate_label(self, y_a, y_b):
        
        y_a, y_b = y_a, y_b
        assert y_a != y_b
        if y_a > y_b:
            tmp = y_a
            y_a = y_b
            y_b = tmp
        label_index = ((2 * self.total_class - y_a - 1) * y_a) / 2 + (y_b - y_a) - 1
        
        return label_index + self.total_class
    
    def train_epoch_intermediate_node_all_class(self, t, trn_loader):
        # if t == 1:
        #     print(c)
        """Runs a single epoch"""
        self.lws_models.eval()
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        
        total_loss_criterion = [] 
        total_loss_mix = [] 
        total_loss_self_sup_lamda = []
        for images, targets in trn_loader:
            # breakpoint()
            images, targets = images.to(self.device), targets.to(self.device)
            org_bs = images.shape[0]
            images, targets = self.classAug(images, targets)
            # org_bs = images.shape[0]
            # breakpoint()
            # mix_images,targets_a, targets_b, lam, index_perm = self.mixup_data(images, targets)
            # images = torch.cat([images, mix_images])

            # breakpoint()
            # Forward current model
            # outputs, features = self.model(images, return_features=True)
            
            outputs, features, int_outputs = self.model.forward_int_nodes(images, return_features=True, int_nodes_output=True)


            
            # breakpoint()
            
            
            lwsoutputs=[]
            ref_outputs = None
            ref_features = None
            # breakpoint()
            if t > 0:
                # print(c)
                # breakpoint()c
                ref_outputs, ref_features = self.ref_model(images, return_features=True)
                # breakpoint()
            loss_criterion = self.criterion(t, outputs, targets, ref_outputs, features, ref_features,lwsoutputs, int_outputs = int_outputs, org_bs=org_bs)
            # loss_mix = self.mixup_criterion(outputs, targets_a, targets_b, lam, org_bs)
            # loss_self_sup_lamda = nn.MSELoss()(self.model.self_sup_head(torch.cat([features[org_bs:],features[:org_bs]], dim=1)).squeeze(-1), torch.tensor(lam).repeat(org_bs).cuda())
            # loss_self_sup_lamda = nn.CrossEntropyLoss(None)(self.model.self_sup_head(features_cat_full), label_self_full.squeeze(-1).long())

            # breakpoint()
            loss = loss_criterion
            # if t==0:
            # loss += loss_mix #+ loss_self_sup_lamda
                # print("adding mix")

            total_loss_criterion.append(loss_criterion.item())
            # total_loss_mix.append(loss_mix.item())
            # total_loss_self_sup_lamda.append(loss_self_sup_lamda.item())
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        print(f'Loss CE: {torch.mean(torch.tensor(total_loss_criterion))},  Loss_mix: {torch.mean(torch.tensor(total_loss_mix))}')
        
    def train_epoch_aug(self, t, trn_loader, trn_loader_aug, label_list, pairs):
        # if t == 1:
        #     print(c)
        """Runs a single epoch"""
        self.lws_models.eval()
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        
        total_loss_criterion = [] 
        total_loss_mix = [] 
        total_loss_self_sup_lamda = []
        for images, targets in trn_loader:
            # breakpoint()
            images, targets = images.to(self.device), targets.to(self.device)
            org_bs = images.shape[0]
            # images, targets = self.classAug(images, targets)
            # org_bs = images.shape[0]
            # breakpoint()
            # mix_images,targets_a, targets_b, lam, index_perm = self.mixup_data(images, targets)
            # images = torch.cat([images, mix_images])

            # breakpoint()
            # Forward current model
            # outputs, features = self.model(images, return_features=True)
            for i1, batch1 in enumerate(trn_loader_aug):
                data1, _ = [_.cuda() for _ in batch1]
                # breakpoint()
                #torchvision.utils.save_image(data1[1],'test.jpg')
                #torchvision.utils.save_image(data1[113],'test1.jpg')
                #print(data1.shape)
                data1 = self.classAug_intermediate(data1, pairs)
                # breakpoint()
                index = torch.randperm(data1.shape[0]).cuda()
                # breakpoint()
                data1 = data1[index,:]
                # breakpoint()
                train_label1 = torch.tensor(label_list).cuda()[index]
                # breakpoint()
                #torchvision.utils.save_image(data1[1],'test2.jpg')
                final_images = torch.cat((images, data1)).cuda()
                #print(final_data.shape)
                final_targets = torch.cat((targets, train_label1)).cuda()
            
            outputs, features, int_outputs = self.model.forward_int_nodes(final_images, return_features=True, int_nodes_output=True)


            
            # breakpoint()
            
            
            lwsoutputs=[]
            ref_outputs = None
            ref_features = None
            # breakpoint()
            if t > 0:
                # print(c)
                # breakpoint()c
                ref_outputs, ref_features = self.ref_model(images, return_features=True)
                # breakpoint()
            loss_criterion = self.criterion(t, outputs, final_targets, ref_outputs, features, ref_features,lwsoutputs, int_outputs = int_outputs, org_bs=org_bs)
            # loss_mix = self.mixup_criterion(outputs, targets_a, targets_b, lam, org_bs)
            # loss_self_sup_lamda = nn.MSELoss()(self.model.self_sup_head(torch.cat([features[org_bs:],features[:org_bs]], dim=1)).squeeze(-1), torch.tensor(lam).repeat(org_bs).cuda())
            # loss_self_sup_lamda = nn.CrossEntropyLoss(None)(self.model.self_sup_head(features_cat_full), label_self_full.squeeze(-1).long())

            # breakpoint()
            loss = loss_criterion
            # if t==0:
            # loss += loss_mix #+ loss_self_sup_lamda
                # print("adding mix")

            total_loss_criterion.append(loss_criterion.item())
            # total_loss_mix.append(loss_mix.item())
            # total_loss_self_sup_lamda.append(loss_self_sup_lamda.item())
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        print(f'Loss CE: {torch.mean(torch.tensor(total_loss_criterion))},  Loss_mix: {torch.mean(torch.tensor(total_loss_mix))}')
        
    def classAug_intermediate(self,data1,pairs,alpha=20.0):
        mix_data = []
        for i in range(len(pairs)):
            lam = np.random.beta(alpha, alpha)
            if lam < 0.4 or lam > 0.6:
                lam = 0.5
            mix_data.append(lam * data1[i] + (1 - lam) * data1[i+len(pairs)])
        
        for item in mix_data:
            data1 = torch.cat((data1, item.unsqueeze(0)), 0)
        return data1[2*len(pairs):]
        
    def train_epoch(self, t, trn_loader):
     
        """Runs a single epoch"""
        self.lws_models.eval()
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        
        total_loss_criterion = [] 
        total_loss_mix = [] 
        total_loss_self_sup_lamda = []
        for images, targets in trn_loader:
            # breakpoint()
            images, targets = images.to(self.device), targets.to(self.device)
            # org_bs = images.shape[0]
            # images, targets = self.classAug(images, targets)
            # org_bs = images.shape[0]
            # breakpoint()
            # mix_images,targets_a, targets_b, lam, index_perm = self.mixup_data(images, targets)
            # images = torch.cat([images, mix_images])

            # breakpoint()
            # Forward current model
            outputs, features = self.model(images, return_features=True)
            
            # outputs, features, int_outputs = self.model.forward_int_nodes(images, return_features=True, int_nodes_output=True)


            
            # breakpoint()
            
            
            lwsoutputs=[]
            ref_outputs = None
            ref_features = None
            # breakpoint()
            if t > 0:
                # print(c)
                # breakpoint()c
                ref_outputs, ref_features = self.ref_model(images, return_features=True)
                # breakpoint()
            # int_outputs=None
            # o
            loss_criterion = self.criterion(t, outputs, targets, ref_outputs, features, ref_features,lwsoutputs)
            # loss_criterion = self.criterion(t, outputs, targets, ref_outputs, features, ref_features,lwsoutputs, int_outputs = int_outputs, org_bs=org_bs)
            # loss_mix = self.mixup_criterion(outputs, targets_a, targets_b, lam, org_bs)
            # loss_self_sup_lamda = nn.MSELoss()(self.model.self_sup_head(torch.cat([features[org_bs:],features[:org_bs]], dim=1)).squeeze(-1), torch.tensor(lam).repeat(org_bs).cuda())
            # loss_self_sup_lamda = nn.CrossEntropyLoss(None)(self.model.self_sup_head(features_cat_full), label_self_full.squeeze(-1).long())

            # breakpoint()
            loss = loss_criterion
            # if t==0:
            # loss += loss_mix #+ loss_self_sup_lamda
                # print("adding mix")

            total_loss_criterion.append(loss_criterion.item())
            # total_loss_mix.append(loss_mix.item())
            # total_loss_self_sup_lamda.append(loss_self_sup_lamda.item())
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        print(f'Loss CE: {torch.mean(torch.tensor(total_loss_criterion))},  Loss_mix: {torch.mean(torch.tensor(total_loss_mix))}')

    def criterion(self, t, outputs, targets, ref_outputs=None, features=None, ref_features=None,stage2 = False,lwsoutputs = [], int_outputs=None, org_bs=None):
        """Returns the loss value"""
        if ref_outputs is None or ref_features is None or stage2:
            # print('stage2')
            if type(outputs[0]) == dict:
                outputs = torch.cat([o['wsigma'] for o in outputs], dim=1)
            else:
                outputs = torch.cat(outputs, dim=1)
            # Eq. 1: regular cross entropy
            # breakpoint()
            if int_outputs is not None:
                # if type(int_outputs[0]) == dict:
                # int_outputs = torch.cat([o['wsigma'] for o in int_outputs], dim=1)
                # else:
                    # int_outputs = torch.cat(int_outputs, dim=1)
                # int_outputs = int_outputs[]
                final_outputs = torch.cat([outputs, int_outputs['wsigma']], dim=1)
                # breakpoint()
                loss = nn.CrossEntropyLoss(None)(final_outputs, targets)
            else:
                loss = nn.CrossEntropyLoss(None)(outputs, targets)
            # breakpoint()
        else:
            if self.less_forget:
                # Eq. 6: Less-Forgetting constraint
                loss_dist = nn.CosineEmbeddingLoss()(features[:targets.shape[0]], ref_features[:targets.shape[0]].detach(),
                                                     torch.ones(targets.shape[0]).to(self.device)) * self.lamda
            else:
                # Scores before scale, [-1, 1]
                ref_outputs = torch.cat([ro['wosigma'] for ro in ref_outputs], dim=1).detach()
                old_scores = torch.cat([o['wosigma'] for o in outputs[:-1]], dim=1)
                num_old_classes = ref_outputs.shape[1]

                # Eq. 5: Modified distillation loss for cosine normalization
                loss_dist = nn.MSELoss()(old_scores, ref_outputs) * self.lamda * num_old_classes

            loss_mr = torch.zeros(1).to(self.device)
            if self.margin_ranking:
                # Scores before scale, [-1, 1]
                outputs_wos = torch.cat([o['wosigma'] for o in outputs], dim=1)
                num_old_classes = outputs_wos.shape[1] - outputs[-1]['wosigma'].shape[1]

                # Sec 3.4: "We select those new classes that yield highest responses to x (...)"
                # The index of hard samples, i.e., samples from old classes
                if org_bs is not None:
                    
                    targets_org = targets[:org_bs]
                else:
                    targets_org = targets
                hard_index = targets_org < num_old_classes
                hard_num = hard_index.sum()
                # breakpoint()

                if hard_num > 0:
                    # Get "ground truth" scores
                    if org_bs is not None:
                        outputs_wos = outputs_wos[:org_bs]
                    else:
                        outputs_wos = outputs_wos
                    
                    # breakpoint()
                    gt_scores = outputs_wos.gather(1, targets_org.unsqueeze(1))[hard_index]
                    gt_scores = gt_scores.repeat(1, self.K)

                    # Get top-K scores on novel classes
                    max_novel_scores = outputs_wos[hard_index, num_old_classes:].topk(self.K, dim=1)[0]

                    assert (gt_scores.size() == max_novel_scores.size())
                    assert (gt_scores.size(0) == hard_num)
                    # Eq. 8: margin ranking loss
                    loss_mr = nn.MarginRankingLoss(margin=self.dist)(gt_scores.view(-1, 1),
                                                                     max_novel_scores.view(-1, 1),
                                                                     torch.ones(hard_num * self.K).to(self.device).view(-1,1))
                    loss_mr *= self.lamb_mr

            # Eq. 1: regular cross entropy
            # loss_ce = nn.CrossEntropyLoss()(torch.cat([o['wsigma'] for o in outputs], dim=1), targets)

            outputs = torch.cat([o['wsigma'] for o in outputs], dim=1)
            if int_outputs is not None:
                # if type(int_outputs[0]) == dict:
                # int_outputs = torch.cat([o['wsigma'] for o in int_outputs], dim=1)
                # else:
                    # int_outputs = torch.cat(int_outputs, dim=1)
                # int_outputs = int_outputs[]
                final_outputs = torch.cat([outputs, int_outputs['wsigma']], dim=1)
                # breakpoint()
                loss_ce = nn.CrossEntropyLoss(None)(final_outputs, targets)
            else:
                loss_ce = nn.CrossEntropyLoss(None)(outputs, targets)
            # loss_ce = nn.CrossEntropyLoss()(outputs[:org_bs], targets) ##### teja
            # Eq. 9: integrated objective
            loss = loss_dist + loss_ce + loss_mr
        return loss

    @staticmethod
    def warmup_luci_loss(outputs, targets):
        if type(outputs) == dict:
            # needed during train
            return torch.nn.functional.cross_entropy(outputs['wosigma'], targets)
        else:
            # needed during eval()
            return torch.nn.functional.cross_entropy(outputs, targets)

# Sec 3.2: This class implements the cosine normalizing linear layer module using Eq. 4
class CosineLinear(Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)  # for initializaiton of sigma

    def forward(self, input, stochastic=None, cs_stoc=None):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out_s = self.sigma * out
        else:
            out_s = out
        if self.training:
            return {'wsigma': out_s, 'wosigma': out}
        else:
            return out_s


# This class implements a ResNet Basic Block without the final ReLu in the forward
class BasicBlockNoRelu(nn.Module):
    expansion = 1

    def __init__(self, conv1, bn1, relu, conv2, bn2, downsample):
        super(BasicBlockNoRelu, self).__init__()
        self.conv1 = conv1
        self.bn1 = bn1
        self.relu = relu
        self.conv2 = conv2
        self.bn2 = bn2
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        # Removed final ReLU
        return out
