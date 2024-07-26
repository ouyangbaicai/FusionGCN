# Automatic Logging and save model with overfitting avoidance.
import os
import datetime

import torch
from Utilities.EarlyStopping import EarlyStopping

class Logging_SaveModel:
    def __init__(self, savepath, hyperparas=None):
        self.SAVEPATH = savepath
        self.start_training_time = datetime.datetime.now().strftime('%Y-%m-%d %H.%M.%S')
        os.mkdir(os.path.join(savepath, self.start_training_time))
        self.ChildDir = os.path.join(savepath, self.start_training_time)
        self.tlf = open(self.ChildDir + '\\' + self.start_training_time + '.txt', 'w')
        # Parse parameter dictionary.
        if hyperparas != None:
            self.tlf.write('HyperParameter:\n')
            self.tlf.write("train_size = " + str(hyperparas['set_size']) + '\n')
            self.tlf.write("valid_size = " + str(hyperparas['set_size']//9) + '\n')
            self.tlf.write("batchsize = " + str(hyperparas['batchsize']) + '\n')
            self.tlf.write("epochs = " + str(hyperparas['epochs']) + '\n')
            self.tlf.write("lr = " + str(hyperparas['lr']) + '\n')
            self.tlf.write("gamma = " + str(hyperparas['gamma']) + '\n')
            self.tlf.write("scheduler_step = " + str(hyperparas['scheduler_step']) + '\n')
            self.tlf.write("lmd = " + str(hyperparas['lmd']) + '\n')
            self.tlf.write("patience = " + str(hyperparas['patience']) + '\n\n')
            self.tlf.write('Training log:\n')
        # To avoid overfitting.
        self.ES = EarlyStopping(os.path.join(os.getcwd(), self.ChildDir), patience=hyperparas['patience'])
        self.ENDTRAIN = False
        print("Trained weights will be saved in the folder: " + os.path.join(os.getcwd(), self.ChildDir) + "\n")

    def __call__(self, model, current_epoch, log_contents, val_loss, save_every_model):
        self.Logging(log_contents)  # Logging
        self.SaveWeights(model, val_loss, current_epoch, save_every_model)  # Save model weights

    def Logging(self, contents):
        self.tlf.write(contents)

    def SaveWeights(self, model, val_loss, current_epoch, save_every_model):
        # To avoid overfitting.
        self.ES(model, val_loss, current_epoch, save_every_model)
        if self.ES.early_stop:
            self.tlf.close()
            self.ENDTRAIN = True



if __name__ == '__main__':
    l = Logging_SaveModel()
    l()