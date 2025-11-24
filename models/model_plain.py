from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam

from models.select_network import define_G
from models.model_base import ModelBase
from models.loss import CharbonnierLoss
from models.loss_ssim import SSIMLoss
from models.network_swinir import SwinIR as TeacherModel

from utils.utils_model import test_mode
from utils.utils_regularizers import regularizer_orth, regularizer_clip


class ModelPlain(ModelBase):
    """Train with pixel loss"""
    def __init__(self, opt):
        super(ModelPlain, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']    # training option
        self.netG = define_G(opt)
        self.netG = self.model_to_device(self.netG)
        if self.opt_train['E_decay'] > 0:
            self.netE = define_G(opt).to(self.device).eval()

        # ---- START OF OUR NEW CODE ----
        #
        #    First, we will check our JSON file for a new "distillation_type" setting.
        #    This is the "switch" that will control our different experiments.
        self.distillation_type = self.opt_train.get('distillation_type', 'none')

        # A. If the switch is not 'none', we need to hire the Teacher.
        if self.distillation_type != 'none':
            print("Knowledge Distillation is active. Loading the Teacher model.")

            # B. Define the Teacher's architecture. These are the parameters for the
            #    large, original SwinIR model. They are hardcoded here.
            teacher_args = {
                'upscale': self.opt['scale'], # Use the same scale as our student
                'in_chans': 3,
                'img_size': 64,
                'window_size': 8,
                'img_range': 1.,
                'depths': [6, 6, 6, 6, 6, 6],
                'embed_dim': 180,
                'num_heads': [6, 6, 6, 6, 6, 6],
                'mlp_ratio': 2,
                'upsampler': 'pixelshuffle',
                'resi_connection': '1conv'
            }
            
            # C. Build the Teacher model from the blueprint and move it to the GPU.
            self.netTeacher = TeacherModel(**teacher_args).to(self.device)
            
            # D. We need to tell the Professor where to find the Teacher's pre-trained brain.
            #    We will add a new setting in our JSON file for this.
            # I am making an assumption that the key in the JSON will be 'pretrained_netTeacher'.
            # Please let me know if you would prefer a different name for this setting.
            load_path_Teacher = self.opt['path']['pretrained_netTeacher']
            
            # E. Load the Teacher's brain and freeze it.
            print(f"Loading Teacher model from: {load_path_Teacher}")
            teacher_state_dict = torch.load(load_path_Teacher)
            self.netTeacher.load_state_dict(teacher_state_dict['params'] if 'params' in teacher_state_dict.keys() else teacher_state_dict, strict=True)
            
            # F. Freeze the Teacher model. This is critical. We are only learning FROM the
            #    teacher, not changing it. This ensures its brain is read-only.
            for param in self.netTeacher.parameters():
                param.requires_grad = False
            
            # G. Put the teacher in "evaluation mode". This also ensures it doesn't change.
            self.netTeacher.eval()
            # H. If we are doing feature distillation, we need to create "translators"
            #    to map the Teacher's large thoughts to the Student's smaller thoughts.
            if self.distillation_type == 'feature':
                print("Creating feature distillation projection layers.")
                
                # We need one translator for each of the Student's "thought" stages.
                # The Student has 4 RSTB blocks.
                num_student_layers = len(self.opt['netG']['depths'])
                
                # The Teacher's language is 180 words. The Student's is 60.
                teacher_dim = 180
                student_dim = self.opt['netG']['embed_dim']
                
                # Create a list of translator layers. Each one is a simple linear network.
                self.projectors = nn.ModuleList()
                for _ in range(num_student_layers):
                    self.projectors.append(
                        nn.Linear(teacher_dim, student_dim).to(self.device)
                    )
        # ---- END OF OUR NEW CODE ----

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.load()                           # load model
        self.netG.train()                     # set training mode,for BN
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.load_optimizers()                # load optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log

    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=self.opt_train['G_param_strict'], param_key='params')
        load_path_E = self.opt['path']['pretrained_netE']
        if self.opt_train['E_decay'] > 0:
            if load_path_E is not None:
                print('Loading model for E [{:s}] ...'.format(load_path_E))
                self.load_network(load_path_E, self.netE, strict=self.opt_train['E_param_strict'], param_key='params_ema')
            else:
                print('Copying model for E ...')
                self.update_E(0)
            self.netE.eval()

    # ----------------------------------------
    # load optimizer
    # ----------------------------------------
    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        if self.opt_train['E_decay'] > 0:
            self.save_network(self.save_dir, self.netE, 'E', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        G_lossfn_type = self.opt_train['G_lossfn_type']
        if G_lossfn_type == 'l1':
            self.G_lossfn = nn.L1Loss().to(self.device)
        elif G_lossfn_type == 'l2':
            self.G_lossfn = nn.MSELoss().to(self.device)
        elif G_lossfn_type == 'l2sum':
            self.G_lossfn = nn.MSELoss(reduction='sum').to(self.device)
        elif G_lossfn_type == 'ssim':
            self.G_lossfn = SSIMLoss().to(self.device)
        elif G_lossfn_type == 'charbonnier':
            self.G_lossfn = CharbonnierLoss(self.opt_train['G_charbonnier_eps']).to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']
        # ---- START OF OUR NEW CODE ----
        #
        # A. Check our "switch". If distillation is active, we need to prepare
        #    a new "exam paper" for it.
        if self.distillation_type != 'none':
            print("Defining the Distillation Loss.")
            
            # B. We will add a new setting to our JSON to control the type of
            #    exam and its importance (weight).
            # I am making an assumption that the keys will be 'distill_lossfn_type'
            # and 'distill_lossfn_weight'. Please let me know if you prefer others.
            distill_loss_type = self.opt_train.get('distill_lossfn_type', 'l1')
            self.distill_loss_weight = self.opt_train.get('distill_lossfn_weight', 1.0)
            
            # C. Define the loss function itself. For standard response-based
            #    distillation, an L1 loss is a very common and effective choice.
            if distill_loss_type == 'l1':
                self.distill_lossfn = nn.L1Loss().to(self.device)
            elif distill_loss_type == 'l2':
                self.distill_lossfn = nn.MSELoss().to(self.device)
            else:
                raise NotImplementedError(f'Distillation Loss type [{distill_loss_type}] is not found.')
        # ---- END OF OUR NEW CODE ----
            # ---- START OF OUR NEW CODE FOR MODEL C ----
            #
            # A. Check if our switch is set to 'feature'. This activates the
            #    "Mind-Reading Exam".
            if self.distillation_type == 'feature':
                print("Defining the Feature Distillation Loss.")
                
                # B. Get the settings for our new exam from the JSON curriculum.
                # I am making an assumption that the keys will be 'feature_lossfn_type'
                # and 'feature_lossfn_weight'.
                feature_loss_type = self.opt_train.get('feature_lossfn_type', 'l1')
                self.feature_loss_weight = self.opt_train.get('feature_lossfn_weight', 1.0)
                
                # C. Define the loss function itself. L1 is a good choice for
                #    comparing the feature maps.
                if feature_loss_type == 'l1':
                    self.feature_lossfn = nn.L1Loss().to(self.device)
                elif feature_loss_type == 'l2':
                    self.feature_lossfn = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError(f'Feature Loss type [{feature_loss_type}] is not found.')
            # ---- END OF OUR NEW CODE FOR MODEL C ----


    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        # ---- START OF OUR FINAL CODE CHANGE ----
        # If we have translators, their parameters must also be trained.
        if self.distillation_type == 'feature':
            for proj in self.projectors:
                G_optim_params.extend(proj.parameters())
        # ---- END OF OUR FINAL CODE CHANGE ----
        if self.opt_train['G_optimizer_type'] == 'adam':
            self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'],
                                    betas=self.opt_train['G_optimizer_betas'],
                                    weight_decay=self.opt_train['G_optimizer_wd'])
        else:
            raise NotImplementedError

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        if self.opt_train['G_scheduler_type'] == 'MultiStepLR':
            self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                            self.opt_train['G_scheduler_milestones'],
                                                            self.opt_train['G_scheduler_gamma']
                                                            ))
        elif self.opt_train['G_scheduler_type'] == 'CosineAnnealingWarmRestarts':
            self.schedulers.append(lr_scheduler.CosineAnnealingWarmRestarts(self.G_optimizer,
                                                            self.opt_train['G_scheduler_periods'],
                                                            self.opt_train['G_scheduler_restart_weights'],
                                                            self.opt_train['G_scheduler_eta_min']
                                                            ))
        else:
            raise NotImplementedError

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        self.L = data['L'].to(self.device)
        if need_H:
            self.H = data['H'].to(self.device)

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self):
        self.E = self.netG(self.L)

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()

        # Step 1: Run the student's forward pass. This single call calculates
        # the final output (self.E) and fills the student's backpack.
        self.netG_forward()

        # Step 2: Initialize the total loss with the standard L1 loss
        # against the ground-truth. This is our base grade.
        l_g_total = self.G_lossfn_weight * self.G_lossfn(self.E, self.H)
        self.log_dict['l_g_L1'] = l_g_total.item()

        # Step 3: If we are doing any kind of distillation...
        if self.distillation_type != 'none':
            
            # Get the Teacher's final answer and its internal "thoughts".
            with torch.no_grad():
                teacher_output = self.netTeacher(self.L)
                teacher_thoughts = self.netTeacher.intermediate_features
            
            # Calculate the response-based distillation loss (used for both Model B and C).
            l_g_distill = self.distill_loss_weight * self.distill_lossfn(self.E, teacher_output)
            l_g_total += l_g_distill
            self.log_dict['l_g_distill'] = l_g_distill.item()

            # Step 4: If we are doing the advanced FEATURE distillation (Model C)...
            if self.distillation_type == 'feature':
                # Get the student's internal "thoughts" from its backpack.
                student_thoughts = self.netG.module.intermediate_features
                
                # Calculate the feature loss using our "translators".
                l_g_feature = torch.tensor(0.0).to(self.device)
                for i in range(len(student_thoughts)):
                    student_thought = student_thoughts[i]
                    
                    # Use the i-th translator to map the Teacher's complex thought
                    # to the Student's simpler thought space before comparing.
                    teacher_thought_translated = self.projectors[i](teacher_thoughts[i])
                    
                    l_g_feature += self.feature_loss_weight * self.feature_lossfn(student_thought, teacher_thought_translated)
                
                l_g_total += l_g_feature
                self.log_dict['l_g_feature'] = l_g_feature.item()

        # Step 5: Backpropagate the final, combined loss.
        l_g_total.backward()

        # The original KAIR code for clipping gradients, stepping the optimizer, etc.
        # This part remains unchanged.
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'], norm_type=2)

        self.G_optimizer.step()

        G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train['G_regularizer_orthstep'] else 0
        if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_orth)
        G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train['G_regularizer_clipstep'] else 0
        if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_clip)

        self.log_dict['G_loss'] = l_g_total.item()

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.netG_forward()
        self.netG.train()

    # ----------------------------------------
    # test / inference x8
    # ----------------------------------------
    def testx8(self):
        self.netG.eval()
        with torch.no_grad():
            self.E = test_mode(self.netG, self.L, mode=3, sf=self.opt['scale'], modulo=1)
        self.netG.train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H image
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float().cpu()
        out_dict['E'] = self.E.detach()[0].float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach()[0].float().cpu()
        return out_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_results(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach().float().cpu()
        out_dict['E'] = self.E.detach().float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach().float().cpu()
        return out_dict

    """
    # ----------------------------------------
    # Information of netG
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg
