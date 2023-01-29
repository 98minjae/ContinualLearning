import torch
from torch.utils import data
from utils.buffer.buffer import Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match
from utils.utils import maybe_cuda, AverageMeter


class ExperienceReplay(ContinualLearner):
    def __init__(self, model, opt, params):
        super(ExperienceReplay, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params) # 버퍼 초기화 및 (업데이트, 추출) 방법 불러오기
        self.mem_size = params.mem_size # 설정: 5000
        self.eps_mem_batch = params.eps_mem_batch # default: 10
        self.mem_iters = params.mem_iters # default: 1

    def train_learner(self, x_train, y_train):
        self.before_train(x_train, y_train) # 현재 테스크 데이터에 대한 레이블 저장 (+레이블 인덱스 딕션어리 생성, +테스크 인덱스 딕션어리 생성)
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
                                       drop_last=True)
        # set up model
        self.model = self.model.train()

        # setup tracker
        losses_batch = AverageMeter() # sum, count 기록
        losses_mem = AverageMeter() # sum, count 기록
        acc_batch = AverageMeter() # sum, count 기록
        acc_mem = AverageMeter() # sum, count 기록

        for ep in range(self.epoch): # default: 1
            for i, batch_data in enumerate(train_loader):
                # batch update
                batch_x, batch_y = batch_data # 학습데이터 배치 불러오기
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                for j in range(self.mem_iters): # default: 1
                    logits = self.model.forward(batch_x)
                    loss = self.criterion(logits, batch_y)
                    if self.params.trick['kd_trick']: # Knowledge distillation with cross entropy trick
                        loss = 1 / (self.task_seen + 1) * loss + (1 - 1 / (self.task_seen + 1)) * \
                                   self.kd_manager.get_kd_loss(logits, batch_x)
                    if self.params.trick['kd_trick_star']: # Improved knowledge distillation trick
                        loss = 1/((self.task_seen + 1) ** 0.5) * loss + \
                               (1 - 1/((self.task_seen + 1) ** 0.5)) * self.kd_manager.get_kd_loss(logits, batch_x)
                    _, pred_label = torch.max(logits, 1) # 예측값
                    correct_cnt = (pred_label == batch_y).sum().item() / batch_y.size(0) # 정답률
                    # update tracker
                    acc_batch.update(correct_cnt, batch_y.size(0)) 
                    losses_batch.update(loss, batch_y.size(0))
                    # backward
                    self.opt.zero_grad()
                    loss.backward()

                    # mem update
                    mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y) # 버퍼에서 사용할 버퍼 추출
                    if mem_x.size(0) > 0:
                        mem_x = maybe_cuda(mem_x, self.cuda)
                        mem_y = maybe_cuda(mem_y, self.cuda)
                        mem_logits = self.model.forward(mem_x)
                        loss_mem = self.criterion(mem_logits, mem_y)
                        if self.params.trick['kd_trick']:
                            loss_mem = 1 / (self.task_seen + 1) * loss_mem + (1 - 1 / (self.task_seen + 1)) * \
                                       self.kd_manager.get_kd_loss(mem_logits, mem_x)
                        if self.params.trick['kd_trick_star']:
                            loss_mem = 1 / ((self.task_seen + 1) ** 0.5) * loss_mem + \
                                   (1 - 1 / ((self.task_seen + 1) ** 0.5)) * self.kd_manager.get_kd_loss(mem_logits,
                                                                                                         mem_x)
                        # update tracker
                        losses_mem.update(loss_mem, mem_y.size(0))
                        _, pred_label = torch.max(mem_logits, 1)
                        correct_cnt = (pred_label == mem_y).sum().item() / mem_y.size(0)
                        acc_mem.update(correct_cnt, mem_y.size(0))

                        loss_mem.backward()

                    if self.params.update == 'ASER' or self.params.retrieve == 'ASER':
                        # opt update
                        self.opt.zero_grad()
                        combined_batch = torch.cat((mem_x, batch_x))
                        combined_labels = torch.cat((mem_y, batch_y))
                        combined_logits = self.model.forward(combined_batch)
                        loss_combined = self.criterion(combined_logits, combined_labels)
                        loss_combined.backward()
                        self.opt.step()
                    else:
                        self.opt.step()

                # update mem
                self.buffer.update(batch_x, batch_y)

                if i % 100 == 1 and self.verbose:
                    print(
                        '==>>> it: {}, avg. loss: {:.6f}, '
                        'running train acc: {:.3f}'
                            .format(i, losses_batch.avg(), acc_batch.avg())
                    )
                    print(
                        '==>>> it: {}, mem avg. loss: {:.6f}, '
                        'running mem acc: {:.3f}'
                            .format(i, losses_mem.avg(), acc_mem.avg())
                    )
        self.after_train()