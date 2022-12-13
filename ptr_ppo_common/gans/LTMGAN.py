import random
import pickle
import torch
from torch import nn
from torch.autograd import Variable 
from torch import autograd
import numpy as np


""" Simple GANS architecture"""




class LTMGAN():
    def __init__(self,dimensions,prev_generator=None):
        self.dimensions = dimensions
        self.old_generator = prev_generator 
        self.latent_dim = 2
        self.batch_size = 64
        self.discriminator = LTMDiscriminator(dimensions)
        self.current_generator = LTMGenerator(self.latent_dim,dimensions)

        self.lr = 1e-4
        self.loss_fn = torch.nn.BCELoss(reduction='sum')
        self.num_epochs = 100
        self.num_task = 1

        self.cuda = True if torch.cuda.is_available else False


        # debugging stuff
        self.g_loss = []
        self.d_loss = []


    def train(self, load_dir=None, load_list=None, num_task=1):
        self.g_loss = []
        self.d_loss = []
        # either supply the directory of a pickle dump or provide a list
        # setup the data
        assert(load_dir != None or load_list != None)

        loss = torch.nn.MSELoss()
        if self.cuda:
            self.current_generator.cuda()
            self.discriminator.cuda()
            loss.cuda()

        optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr= self.lr)
        optimizer_generator = torch.optim.Adam(self.current_generator.parameters(), lr= self.lr)
        
        Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        #with autograd.detect_anomaly():

        valid_labels = Variable(Tensor(self.batch_size, 1).fill_(1.0), requires_grad=False)
        fake_labels = Variable(Tensor(self.batch_size, 1).fill_(0.0), requires_grad=False)

        for epoch in range(self.num_epochs):
            # get the real sample from either the actual data or previously generated data
            samples = training_samples(load_list,self.num_task,300,self.batch_size)
            for real_samples in samples:
                # train generator

                optimizer_generator.zero_grad()
                z = Variable(Tensor(np.random.normal(0, 1, (self.batch_size, self.latent_dim))))
                generated_samples = self.current_generator(z) 
                generator_loss = loss(self.discriminator(generated_samples), valid_labels)
                
                generator_loss.backward()
                optimizer_generator.step()

                # train discriminator
                optimizer_discriminator.zero_grad()
                real_sample_loss = loss(self.discriminator(real_samples), valid_labels)
                generated_sample_loss = loss(self.discriminator(generated_samples.detach()), fake_labels)
                discriminator_loss = (real_sample_loss + generated_sample_loss) / 2

                discriminator_loss.backward()
                optimizer_discriminator.step()
                
                self.g_loss.append(generator_loss.item())
                self.d_loss.append(discriminator_loss.item())
                print("Epoch: %d G: %f  D: %f" % (epoch,generator_loss.item(), discriminator_loss.item()))


    # def compute_gradient_penalty(self, real, generated):
    #     Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
    #     alpha = Tensor(np.random.random((real.size(0), 1)))
    #     interpolates = (alpha * real + ((1-alpha) * generated)).requires_grad_(True)
    #     d_interpolates = self.discriminator(interpolates)

    #     fake = Variable(Tensor(real.shape[0], 1).fill_(1.0), requires_grad = False)

    #     gradients = autograd.grad(
    #         outputs=d_interpolates,
    #         inputs=interpolates,
    #         grad_outputs=fake,
    #         create_graph=True,
    #         retain_graph=True,
    #         only_inputs=True,
    #     )[0]

    #     gradients = gradients.view(gradients.size(0), -1)
    #     gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    #     return gradient_penalty



    # def wgan_train(self,load_list):
    #     self.g_loss = []
    #     self.d_loss = []
    #     lambda_gp = 10
    #     Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
    #     if self.cuda:
    #         self.current_generator.cuda()
    #         self.discriminator.cuda()

    #     optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr= self.lr)
    #     optimizer_generator = torch.optim.Adam(self.current_generator.parameters(), lr= self.lr)

    #     for epoch in range(self.num_epochs):
    #         samples = training_samples(load_list,self.num_task,100,self.batch_size)
    #         i = 0
    #         for real_samples in samples:
    #             # train discriminator
    #             optimizer_discriminator.zero_grad()
    #             z = Variable(Tensor(np.random.normal(0, 1, (self.batch_size, self.latent_dim))))
    #             generated_samples = self.current_generator(z)
                
    #             real_d = self.discriminator(real_samples)
    #             fake_d = self.discriminator(generated_samples)
    #             gradient_d = self.compute_gradient_penalty(real_samples, generated_samples)
    #             d_loss = -torch.mean(real_d) + torch.mean(fake_d) + lambda_gp * gradient_d
    #             d_loss.backward()
    #             optimizer_discriminator.step()

    #             optimizer_generator.zero_grad()
                
    #             if i % 5 == 0:
    #                 generated_samples = self.current_generator(z)
    #                 fake_g  = self.discriminator(generated_samples)
    #                 g_loss = -torch.mean(fake_g)

    #                 g_loss.backward()
    #                 optimizer_generator.step()
    #                 self.g_loss.append(g_loss.item())
    #                 self.d_loss.append(d_loss.item())
    #                 print("Epoch: %d G: %f  D: %f" % (epoch,g_loss.item(), d_loss.item()))
    #             i += 1






    def update_generators(self):
        self.old_generator = self.current_generator
        self.current_generator = LTMGenerator(self.dimensions)


class training_samples():
    def __init__(self, data, task_size, n,batch_size, old_generator=None):
        self.generator = old_generator
        self.task_size = task_size
        self.data = data
        self.length = n
        self.index = 0
        self.batch_size=batch_size
    
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()
    
    def next(self):
        if self.index < self.length:
            self.index += 1
            if random.random() < (1/self.task_size) or self.old_generator is None:
                # sample from actual data
                z = torch.randperm(len(self.data))[:self.batch_size]
                return self.data[z]
            else:
                # generate previous data
                z = Variable(torch.FloatTensor(np.random.normal(0,1,(self.batch_size, self.generator.dimension))))
                return self.generator(z)
        raise StopIteration()
        # if self.index < len(self.batch_list):
        #     self.index += 1
        #     if len(self.batch_list[self.index]) < self.batch_size:
        #         self.index += 1
        #     return self.batch_list[self.index % len(self.batch_list)]
        # raise StopIteration()

class LTMDiscriminator(nn.Module):
    def __init__(self,inputsize):
        super().__init__()
        # self.model = nn.Sequential(
        #     nn.Linear(inputsize,128),
        #     nn.LeakyReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(128,64),
        #     nn.LeakyReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(64,32),
        #     nn.LeakyReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(32,1),
        #     nn.Tanh(),
        # )

        self.model = nn.Sequential(
            nn.Linear(inputsize, 512), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        output = self.model(x)
        return output

class LTMGenerator(nn.Module):
    def __init__(self,size,output):
        super().__init__()
        self.dimension=size
        self.model = nn.Sequential(
            nn.Linear(size,128),
            nn.BatchNorm1d(128,momentum=.9,eps=1e-5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128,256),
            nn.BatchNorm1d(256,momentum=.9,eps=1e-5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256,512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, output),
            #nn.Tanh(),
        )
    def forward(self,x):
        output = self.model(x)
        return output
    



 



