import torch
import logging
import utils

class StyleTransferTrainer:

    def __init__(self, trainable_net, fixed_net, train_loader, style_image_gram_matrices, mse_loss, optimizer,
                 CONTENT_WEIGHT, STYLE_WEIGHT, TV_REG_WEIGHT,
                 log_level= logging.INFO, device='cpu' ):

        self.trainable_net = trainable_net
        self.fixed_net = fixed_net
        self.train_loader = train_loader
        self.device = device
        self.style_image_gram_matrices = style_image_gram_matrices
        self.mse_loss = mse_loss
        self.optimizer = optimizer
        self.CONTENT_WEIGHT = CONTENT_WEIGHT
        self.STYLE_WEIGHT   = STYLE_WEIGHT
        self.TV_REG_WEIGHT  = TV_REG_WEIGHT
        self.log_level = log_level

    def train(self, epochs = 100, starts = 0, save_model_every_n_epochs = 10, save_to_dir = './saved_model'):
        """

        :param epochs:
        :param settarts:
        :param save_model_every_n_epochs:
        :param save_to_dir:
        :return:

        saves  model
        saves total and component losses in pickle files
        plots every 10 epochs
        """

        logger_ = logging.getLogger()
        logger_.setLevel(self.log_level)

        logger_.info("Training begins")
        self.trainable_net.train()
        steps=0
        while steps <= epochs:

            logger_.info(f"Epoch {steps}")

            total_content_loss = 0.
            total_style_loss = 0.
            total_tv_loss = 0.
            for x,_ in self.train_loader:
                
                
                self.optimizer.zero_grad()
                
                x = x.to(self.device)
                
                
                y = self.trainable_net(x)
                
                
                y_ = self.fixed_net(y)  #embeddings from vgg ->  ("relu1_2", "relu2_2", "relu3_3", "relu4_3")
                
                
                y_content_embedding = y_[2]
                   
                
                y_style_embedding = y_[:]

                with torch.no_grad():
                    content_x = x.detach()

                content_x_vgg = self.fixed_net(content_x)[2]
                
                content_loss = self.mse_loss(content_x_vgg, y_content_embedding)
                

                y_style_gram_matrices = list(map(utils.gram_matrix, y_style_embedding))
                
                
                
                style_mse_losses =  torch.tensor([ self.mse_loss(i,j.expand_as(i)) for i,j in zip(y_style_gram_matrices, self.style_image_gram_matrices)])
                tv_loss = self.TV_REG_WEIGHT * (torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) +
                                                torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))
                total_content_loss += content_loss
                total_style_loss += torch.sum(style_mse_losses)
                total_tv_loss += tv_loss

            total_content_loss *= self.CONTENT_WEIGHT
            total_style_loss *= self.STYLE_WEIGHT
            total_tv_loss *= self.TV_REG_WEIGHT
            total_loss = torch.tensor(total_content_loss + total_style_loss + total_tv_loss)
            total_loss.backward()   # read autograd part before debugging
            self.optimizer.step()


            #append to dictionary for losses, with epoch number as key - loss should be NamedTuple

            if steps % save_model_every_n_epochs == 0:
                # save model
                # plot model so far # call plot function
                pass

        return


if __name__ == '__main__':

    from real_time_style_transfer import get_Image_Transform_Network,get_VGG_network,utils
    test_padding_tfnet = get_Image_Transform_Network.RTST_ImgTfNetPadding()
    test_vgg_network = get_VGG_network.VGGNetwork()
    test_x = torch.rand(1, 3, 256, 256)

    test_trainer = StyleTransferTrainer(test_padding_tfnet, test_vgg_network, train_loader, style_image_gram_matrices, mse_loss, optimizer,
                 1, 1, 1,
                 log_level= logging.INFO, device='cpu' )