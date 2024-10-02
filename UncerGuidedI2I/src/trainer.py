import torch
from losses import *
from networks import *
import random
import time

import torch.nn.functional as F
import torch.optim

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def train_I2I_CasUNetGAN( #MedGAN
    netG_A,
    netD_A,
    train_loader, val_loader,
    dtype=torch.cuda.FloatTensor,
    device='cuda',
    num_epochs=50,
    init_lr=1e-4,
    ckpt_path='../ckpt/I2I_CasUNetGAN',
):
    netG_A.to(device)
    netG_A.type(dtype)
    ####
    netD_A.to(device)
    netD_A.type(dtype)
    ####
    optimizerG = torch.optim.Adam(list(netG_A.parameters()), lr=init_lr)
    optimizerD = torch.optim.Adam(list(netD_A.parameters()), lr=init_lr)
    optimG_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, num_epochs)
    optimD_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, num_epochs)
    ####

    best_mae = 1e9
    # best_mape = 1e9

    for eph in range(num_epochs):
        netG_A.train()
        netD_A.train()
        print('Start epoch {}'.format(eph))
        start_time = time.time()
        for i, batch in enumerate(train_loader):
            if i > 50: 
                break
            xA, xB = batch[0].to(device).type(dtype), batch[1].to(device).type(dtype)
            #calc all the required outputs
            rec_B = netG_A(xA)
            #first gen
            netD_A.eval()
            total_loss = F.l1_loss(rec_B, xB)
            t0 = netD_A(rec_B)
            t1 = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
            e5 = 0.001*F.mse_loss(t1, torch.ones(t1.size()).to(device).type(dtype))
            total_loss += e5
            optimizerG.zero_grad()
            total_loss.backward()
            optimizerG.step()
            optimG_scheduler.step()

            #then discriminator
            netD_A.train()
            t0 = netD_A(xB)
            pred_real_A = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
            loss_D_A_real = 1*F.mse_loss(
                pred_real_A, torch.ones(pred_real_A.size()).to(device).type(dtype)
            )
            t0 = netD_A(rec_B.detach())
            pred_fake_A = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
            loss_D_A_pred = 1*F.mse_loss(
                pred_fake_A, torch.zeros(pred_fake_A.size()).to(device).type(dtype)
            )
            loss_D_A = (loss_D_A_real + loss_D_A_pred)*0.5

            loss_D = loss_D_A
            optimizerD.zero_grad()
            loss_D.backward()
            optimizerD.step()
            optimD_scheduler.step()

            if i % 10 == 0:
                print(
                    'Batch: [{}/{}] | cur_d_loss: {} | cur_tot_loss: {}'.format(
                        i, len(train_loader), loss_D.item(), total_loss.item()
                    )
                )
        end_time = time.time()
        
        print("Training per epoch: {}".format(end_time - start_time))
        
        netG_A.eval()
        netD_A.eval()
        avg_mae = []
        # avg_mape = []
        print('Validating...')
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                xA, xB = batch[0].type(dtype), batch[1].type(dtype)
                
                rec_B = netG_A(xA)
                
                n = xB.shape[0]
                
                for t in range(n):
                    pet_pred = rec_B[t].squeeze(0).cpu().numpy()
                    pet_gt = xB[t].squeeze(0).cpu().numpy()
                    
                    mae = compute_mae(pet_gt, pet_pred)
                    # mape = compute_mape(1 - pet_gt, 1 - pet_pred)
                
                    avg_mae.append(mae)
                    # avg_mape.append(mape)
            
        avg_mae = np.mean(avg_mae)
        # avg_mape = np.mean(avg_mape)

        print(
            'Epoch: [{}/{}] | avg_mae: {}'.format(
                eph, num_epochs, avg_mae
            )
        )
        
        # print(
        #     'Epoch: [{}/{}] | avg_mae: {} | avg_mape: {}'.format(
        #         eph, num_epochs, avg_mae, avg_mape
        #     )
        # )

        if avg_mae < best_mae and avg_mae * 32767 > 310: 
            # remove_file(ckpt_path+'_G_best_mae_{}.pth'.format(best_mae))
            # remove_file(ckpt_path+'_D_best_mae_{}.pth'.format(best_mae))
            
            best_mae = avg_mae
            
            torch.save(netG_A.state_dict(), ckpt_path+'_G_best_mae_{}.pth'.format(best_mae))
            # torch.save(netD_A.state_dict(), ckpt_path+'_D_best_mae_{}.pth'.format(best_mae))
        
        # if avg_mape < best_mape: 
        #     remove_file(ckpt_path+'_G_best_mape_{}.pth'.format(best_mape))
        #     remove_file(ckpt_path+'_D_best_mape_{}.pth'.format(best_mape))
            
        #     best_mape = avg_mape
            
        #     torch.save(netG_A.state_dict(), ckpt_path+'_G_best_mape_{}.pth'.format(best_mape))
        #     torch.save(netD_A.state_dict(), ckpt_path+'_D_best_mape_{}.pth'.format(best_mape))

    return netG_A, netD_A

def train_I2I_CasUNet3headGAN(
    netG_A,
    netD_A,
    train_loader, val_loader,
    dtype=torch.cuda.FloatTensor,
    device='cuda',
    num_epochs=50,
    init_lr=1e-4,
    ckpt_path='../ckpt/I2I_CasUNet3headGAN',
):
    netG_A.to(device)
    netG_A.type(dtype)
    ####
    netD_A.to(device)
    netD_A.type(dtype)
    ####
    optimizerG = torch.optim.Adam(list(netG_A.parameters()), lr=init_lr)
    optimizerD = torch.optim.Adam(list(netD_A.parameters()), lr=init_lr)
    optimG_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, num_epochs)
    optimD_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, num_epochs)
    ####
    list_num_epochs = [num_epochs, num_epochs, 100 + num_epochs]
    list_lambda1 = [1, 0.5, 0.1]
    list_lambda2 = [0.0001, 0.001, 0.01]

    best_mae = 1e9
    # best_mape = 1e9

    for num_epochs, lam1, lam2 in zip(list_num_epochs, list_lambda1, list_lambda2):
        for eph in range(num_epochs):
            netG_A.train()
            netD_A.train()
            print('Start epoch {}'.format(eph))
            for i, batch in enumerate(train_loader):
                if i > 1000:
                    break
                
                xA, xB = batch[0].to(device).type(dtype), batch[1].to(device).type(dtype)
                #calc all the required outputs
                rec_B, rec_alpha_B, rec_beta_B = netG_A(xA)
                #first gen
                netD_A.eval()
                total_loss = lam1*F.l1_loss(rec_B, xB) + lam2*bayeGen_loss(rec_B, rec_alpha_B, rec_beta_B, xB)
                t0 = netD_A(rec_B)
                t1 = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
                e5 = 0.001*F.mse_loss(t1, torch.ones(t1.size()).to(device).type(dtype))
                total_loss += e5
                optimizerG.zero_grad()
                total_loss.backward()
                optimizerG.step()
                optimG_scheduler.step()

                #then discriminator
                netD_A.train()
                t0 = netD_A(xB)
                pred_real_A = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
                loss_D_A_real = 1*F.mse_loss(
                    pred_real_A, torch.ones(pred_real_A.size()).to(device).type(dtype)
                )
                t0 = netD_A(rec_B.detach())
                pred_fake_A = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
                loss_D_A_pred = 1*F.mse_loss(
                    pred_fake_A, torch.zeros(pred_fake_A.size()).to(device).type(dtype)
                )
                loss_D_A = (loss_D_A_real + loss_D_A_pred)*0.5

                loss_D = loss_D_A
                optimizerD.zero_grad()
                loss_D.backward()
                optimizerD.step()
                optimD_scheduler.step()

                if i % 10 == 0:
                    print(
                        'Batch: [{}/{}] | cur_d_loss: {} | cur_tot_loss: {}'.format(
                            i, min(len(train_loader), 1000), loss_D.item(), total_loss.item()
                        )
                    )

            netG_A.eval()
            netD_A.eval()
            avg_mae = []
            # avg_mape = []
            print('Validating...')
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    xA, xB = batch[0].type(dtype), batch[1].type(dtype)
                   
                    rec_B, rec_alpha_B, rec_beta_B = netG_A(xA)
                    
                    n = xB.shape[0]
                    
                    for t in range(n):
                        pet_pred = rec_B[t].cpu().numpy()
                        pet_gt = xB[t].cpu().numpy()
                    
                        mae = compute_mae(pet_gt, pet_pred)
                        # mape = compute_mape(1 - pet_gt, 1 - pet_pred)
                    
                        avg_mae.append(mae)
                        # avg_mape.append(mape)
                
            avg_mae = np.mean(avg_mae)
            # avg_mape = np.mean(avg_mape)

            print(
                'Epoch: [{}/{}] | avg_mae: {}'.format(
                    eph, num_epochs, avg_mae
                )
            )

            # print(
            #     'Epoch: [{}/{}] | avg_mae: {} | avg_mape: {}'.format(
            #         eph, num_epochs, avg_mae, avg_mape
            #     )
            # )

            if avg_mae < best_mae and avg_mae * 32767 > 335: 
                remove_file(ckpt_path+'_G_best_mae_{}.pth'.format(best_mae))
                remove_file(ckpt_path+'_D_best_mae_{}.pth'.format(best_mae))
                
                best_mae = avg_mae
                
                torch.save(netG_A.state_dict(), ckpt_path+'_G_best_mae_{}.pth'.format(best_mae))
                torch.save(netD_A.state_dict(), ckpt_path+'_D_best_mae_{}.pth'.format(best_mae))
            
            # if avg_mape < best_mape: 
            #     remove_file(ckpt_path+'_G_best_mape_{}.pth'.format(best_mape))
            #     remove_file(ckpt_path+'_D_best_mape_{}.pth'.format(best_mape))
                
            #     best_mape = avg_mape
                
            #     torch.save(netG_A.state_dict(), ckpt_path+'_G_best_mape_{}.pth'.format(best_mape))
            #     torch.save(netD_A.state_dict(), ckpt_path+'_D_best_mape_{}.pth'.format(best_mape))

    return netG_A, netD_A

def train_I2I_Sequence_CasUNet3headGAN(
    list_netG_A,
    list_netD_A,
    train_loader, val_loader,
    dtype=torch.cuda.FloatTensor,
    device='cuda',
    num_epochs=50,
    init_lr=1e-4,
    ckpt_path='../ckpt/I2I_Sequence_CasUNet3headGAN',
):
    for nid, m1 in enumerate(list_netG_A):
        m1.to(device)
        m1.type(dtype)
        list_netG_A[nid] = m1
        
    for nid, m2 in enumerate(list_netD_A):
        m2.to(device)
        m2.type(dtype)
        list_netD_A[nid] = m2
    ####
    optimizerG = torch.optim.Adam(list(list_netG_A[-1].parameters()), lr=init_lr)
    optimizerD = torch.optim.Adam(list(list_netD_A[-1].parameters()), lr=init_lr)
    optimG_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, num_epochs)
    optimD_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, num_epochs)
    ####
    list_epochs = [num_epochs, num_epochs, 100 + num_epochs]
    list_lambda1 = [1, 0.5, 0.1]
    list_lambda2 = [0.0001, 0.001, 0.01]
    netG_A, netD_A = list_netG_A[-1], list_netD_A[-1]
    ####

    best_mae = 1e9
    # best_mape = 1e9

    for num_epochs, lam1, lam2 in zip(list_epochs, list_lambda1, list_lambda2):
        for eph in range(num_epochs):
            netG_A.train()
            netD_A.train()
            print('Start epoch {}'.format(eph))
            for i, batch in enumerate(train_loader):
                if i > 900:
                    break
                
                xA, xB = batch[0].to(device).type(dtype), batch[1].to(device).type(dtype)
                #calc all the required outputs
                
                for nid, netG in enumerate(list_netG_A):
                    if nid == 0:
                        rec_B, rec_alpha_B, rec_beta_B = netG(xA)
                    else:
                        xch = torch.cat([rec_B, rec_alpha_B, rec_beta_B, xA], dim=1)
                        rec_B, rec_alpha_B, rec_beta_B = netG(xch)

                #first gen
                netD_A.eval()
                total_loss = lam1*F.l1_loss(rec_B, xB) + lam2*bayeGen_loss(rec_B, rec_alpha_B, rec_beta_B, xB)
                t0 = netD_A(rec_B)
                t1 = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
                e5 = 0.001*F.mse_loss(t1, torch.ones(t1.size()).to(device).type(dtype))
                total_loss += e5
                optimizerG.zero_grad()
                total_loss.backward()
                optimizerG.step()
                optimG_scheduler.step()

                #then discriminator
                netD_A.train()
                t0 = netD_A(xB)
                pred_real_A = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
                loss_D_A_real = 1*F.mse_loss(
                    pred_real_A, torch.ones(pred_real_A.size()).to(device).type(dtype)
                )
                t0 = netD_A(rec_B.detach())
                pred_fake_A = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
                loss_D_A_pred = 1*F.mse_loss(
                    pred_fake_A, torch.zeros(pred_fake_A.size()).to(device).type(dtype)
                )
                loss_D_A = (loss_D_A_real + loss_D_A_pred)*0.5

                loss_D = loss_D_A
                optimizerD.zero_grad()
                loss_D.backward()
                optimizerD.step()
                optimD_scheduler.step()

                if i % 10 == 0:
                    print(
                        'Batch: [{}/{}] | cur_d_loss: {} | cur_tot_loss: {}'.format(
                            i, min(len(train_loader), 1000), loss_D.item(), total_loss.item()
                        )
                    )

            netG_A.eval()
            netD_A.eval()
            avg_mae = []
            # avg_mape = []
            print('validating...')
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    xA, xB = batch[0].to(device).type(dtype), batch[1].to(device).type(dtype)
                
                    for nid, netG in enumerate(list_netG_A):
                        if nid == 0:
                            rec_B, rec_alpha_B, rec_beta_B = netG(xA)
                        else:
                            xch = torch.cat([rec_B, rec_alpha_B, rec_beta_B, xA], dim=1)
                            rec_B, rec_alpha_B, rec_beta_B = netG(xch)

                    n = xB.shape[0]
                    
                    for t in range(n):
                        pet_pred = rec_B[t].squeeze(0).cpu().numpy()
                        pet_gt = xB[t].squeeze(0).cpu().numpy()

                        mae = compute_mae(pet_gt, pet_pred)
                        # mape = compute_mape(1 - pet_gt, 1 - pet_pred)
                    
                        avg_mae.append(mae)
                        # avg_mape.append(mape)
                
            avg_mae = np.mean(avg_mae)
            # avg_mape = np.mean(avg_mape)

            print(
                'Epoch: [{}/{}] | avg_mae: {}'.format(
                    eph, num_epochs, avg_mae
                )
            )

            # print(
            #     'Epoch: [{}/{}] | avg_mae: {} | avg_mape: {}'.format(
            #         eph, num_epochs, avg_mae, avg_mape
            #     )
            # )

            if avg_mae < best_mae and avg_mae * 32767 > 335: 
                # remove_file(ckpt_path+'_G_best_mae_{}.pth'.format(best_mae))
                # remove_file(ckpt_path+'_D_best_mae_{}.pth'.format(best_mae))
                
                best_mae = avg_mae
                
                torch.save(netG_A.state_dict(), ckpt_path+'_G_best_mae_{}.pth'.format(best_mae))
                # torch.save(netD_A.state_dict(), ckpt_path+'_D_best_mae_{}.pth'.format(best_mae))
            
            # if avg_mape < best_mape: 
            #     remove_file(ckpt_path+'_G_best_mape_{}.pth'.format(best_mape))
            #     remove_file(ckpt_path+'_D_best_mape_{}.pth'.format(best_mape))
                
            #     best_mape = avg_mape
                
            #     torch.save(netG_A.state_dict(), ckpt_path+'_G_best_mape_{}.pth'.format(best_mape))
            #     torch.save(netD_A.state_dict(), ckpt_path+'_D_best_mape_{}.pth'.format(best_mape))

    return list_netG_A, list_netD_A