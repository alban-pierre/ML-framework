import torch
import torchvision
import torch.nn as nn
import torch.nn.modules as M
import copy
from math import log10, floor
import time
import numpy as np

from .visualization_utils import *



def train(net, trainloader, criterion, optimizer, nbr_epochs=-1, nbr_images=-1, max_time=-1, train_monitors=[], resume=None):
    total_time = time.time()
    try:
        if (nbr_epochs < 0) and (nbr_images < 0):
            print("Warning : you should set either nbr_epochs or nbr_images")
            nbr_epochs = 1
        if (nbr_epochs < 0):
            nbr_epochs = 2 + int(nbr_images/len(trainloader)) # if we arrive at the end of an epoch
        if isinstance(train_monitors, Train_Monitor):
            train_monitors = [train_monitors]
        if isinstance(train_monitors, list):
            train_monitors = List_Train_Monitor(train_monitors, default_step=len(trainloader))
        # Initialization and/or loading of the previous train parameters
        if resume is None:
            resume = {}
        train_state = {}
        # We define monitors data
        monitors_data = {}
        for k in train_monitors.names():
            monitors_data[k] = []
            if k in resume.keys():
                monitors_data[k] = resume[k]
        # We setup the train state
        total_running_loss = 0.0
        if ("total_running_loss" in resume.keys()):
            total_running_loss = resume["total_running_loss"]
        total_train_error = 0
        if ("total_train_error" in resume.keys()):
            total_train_error = resume["total_train_error"]
        nbr_loss_examples = 0
        if ("nbr_loss_examples" in resume.keys()):
            nbr_loss_examples = resume["nbr_loss_examples"]
        nbr_train_examples = 0
        if ("nbr_train_examples" in resume.keys()):
            nbr_train_examples = resume["nbr_train_examples"]
        start_epoch = 0
        if ("epoch" in resume.keys()):
            start_epoch = resume["epoch"]
        i = 0
        if ("i" in resume.keys()):
            i = resume["i"]
        start_i = i
        stop_time = time.time()
        start_time = stop_time
        if ("time" in resume.keys()):
            start_time = stop_time - resume["time"]
        urgent_stop = False
        # We initialize the train state variable
        train_state["i"] = i
        train_state["epoch"] = int((i-1)/len(trainloader))
        train_state["time"] = stop_time - start_time
        train_state["total_running_loss"] = total_running_loss
        train_state["total_train_error"] = total_train_error
        train_state["nbr_loss_examples"] = nbr_loss_examples
        train_state["nbr_train_examples"] = nbr_train_examples
        train_monitors.resume(net, train_state)
        # We start the clock just before training !
        start_time += time.time() - stop_time
        for epoch in range(start_epoch, start_epoch + nbr_epochs):
            if urgent_stop:
                break;
            for data in trainloader:
                if (nbr_images >= 0) and (i >= nbr_images + start_i):
                    urgent_stop = True
                    break;
                if (max_time >= 0) and (time.time() - start_time > max_time):
                    urgent_stop = True
                    break;
                i += 1
                # get the inputs
                inputs, labels = data
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # print statistics
                to_print = ""
                # Update running loss
                total_running_loss += loss.item()
                nbr_loss_examples += 1
                # Update train error
                _, predicted = torch.max(outputs.data, 1)
                total_train_error += (predicted != labels).sum().item()
                nbr_train_examples += labels.size(0)
                # We execute the train monitors
                stop_time = time.time()
                train_state["i"] = i
                train_state["epoch"] = int((i-1)/len(trainloader))
                train_state["time"] = stop_time - start_time
                train_state["total_running_loss"] = total_running_loss
                train_state["total_train_error"] = total_train_error
                train_state["nbr_loss_examples"] = nbr_loss_examples
                train_state["nbr_train_examples"] = nbr_train_examples
                monitors_data, to_print = train_monitors(net, train_state, monitors_data)
                # for k,v in res.items():
                #     monitors_data[k].append((i, stop_time - start_time, v))
                start_time += time.time() - stop_time
                # We print statistics
                if (to_print != ""):
                    to_print = "i : {0: <10}".format(i) + to_print
                    to_print = "epoch : {0: <5}".format(int((i-1)/len(trainloader))) + to_print
                    to_print = "time : {0: <10}".format(round(time.time() - start_time, 3)) + to_print
                    print(to_print)
        # We save the results, also used for resuming training later
        for k in train_state.keys():
            resume[k] = train_state[k]
        for k in monitors_data.keys():
            resume[k] = monitors_data[k]
        print("\n Total time of the train : {} s".format(round(time.time() - total_time, 3)))
        return resume
    except KeyboardInterrupt:
        train_state["i"] = i
        train_state["epoch"] = int((i-1)/len(trainloader))
        train_state["time"] = stop_time - start_time
        train_state["total_running_loss"] = total_running_loss
        train_state["total_train_error"] = total_train_error
        train_state["nbr_loss_examples"] = nbr_loss_examples
        train_state["nbr_train_examples"] = nbr_train_examples
        for k in train_state.keys():
            resume[k] = train_state[k]
        for k in monitors_data.keys():
            resume[k] = monitors_data[k]
        print("\n Total time of the train : {} s".format(round(time.time() - total_time, 3)))
        raise



def test_output(net, testloader):
    all_predicted = []
    all_truth = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            all_predicted += list(predicted.detach().numpy())
            all_truth += list(labels.detach().numpy())
    return (np.asarray(all_predicted), np.asarray(all_truth))



def prediction_error(predicted, ground_truth, to_print=False):
    ne = np.sum((predicted != ground_truth).astype(int))
    pred_error = ne*100./predicted.shape[0]
    if to_print:
        print("error = {}".format(round(pred_error, 3)))
    return pred_error



def prediction_error_per_class(predicted, ground_truth):
    pred_errors = []
    for i in range(np.max(ground_truth) + 1):
        gt_eq_i = (ground_truth == i).astype(int)
        ne = np.sum((predicted != i).astype(int)*gt_eq_i)
        pred_errors.append(ne*100./np.sum(gt_eq_i))
    return pred_errors



def prediction_false_positive_per_class(predicted, ground_truth):
    pred_errors = []
    for i in range(np.max(ground_truth) + 1):
        gt_neq_i = (ground_truth != i).astype(int)
        ne = np.sum((predicted == i).astype(int)*gt_neq_i)
        pred_errors.append(ne*100./np.sum(1-gt_neq_i))
    return pred_errors



def test(net, testloader, to_print=True):
    predicted, ground_truth = test_output(net, testloader)
    test_error = prediction_error(predicted, ground_truth, to_print=False)
    if to_print:
         print("test_error = {}".format(round(test_error, 3)))
    return test_error



def full_test(net, testloader, classes, to_print=True):
    """ Deprecated """
    nbr_classes = len(classes)
    class_not_correct = [0 for i in range(nbr_classes)]
    class_total = [0 for i in range(nbr_classes)]
    class_fp = [0 for i in range(nbr_classes)]
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted != labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_not_correct[label] += c[i].item()
                class_total[label] += 1
                class_fp[predicted[i]] += c[i].item()
    if to_print:
        test_error = np.sum(class_not_correct) * 100. / np.sum(class_total)
        s = "average error = {}\n".format(round(test_error, 3))
        lc = np.max([len(c) for c in classes])
        tc = "{{0: >{}}} : ".format(lc)
        for i, c in enumerate(classes):
            if (class_total[i] == 0):
                class_total[i] = -1
            s2 = tc.format(c)
            s2 += "error = {0: <10}".format(round(class_not_correct[i] * 100. / class_total[i], 3))
            s2 += "fp = {0: <10}".format(round(class_fp[i] * 100. / class_total[i], 3))
            s += s2 + '\n'
        print(s)
    res = {}
    res["error"] = class_not_correct
    res["total"] = class_total
    res["fp"] = class_fp
    return res



def get_layers(net):
    return _get_layers(net, [])



def _get_layers(net, all_layers=[]):
    for layer in net.children():
        if type(layer) == nn.Sequential: # if sequential layer
            all_layers += _get_layers(layer, [])
        if list(layer.children()) == []: # if leaf node, add it to list
            all_layers.append(layer)
    return all_layers



def freeze(net, layers=3):
    all_layers = get_layers(net)
    all_layers = [lay for lay in all_layers if hasattr(lay, 'weight')]
    if (layers < 0):
        layers += len(all_layers)
    for i,lay in enumerate(all_layers):
        if (i < layers):
            lay.weight.requires_grad = False
            lay.bias.requires_grad = False
        else:
            lay.weight.requires_grad = True
            lay.bias.requires_grad = True


# ------------------------------------------------------------------ #
# |                                                                  |
# |     Now we define functions that will be called inside train     |
# |                                                                  |
# ------------------------------------------------------------------ #



class Train_Monitor(object):

    def __init__(self, step=None):
        self.name = "base_class"
        self.step=step

    def resume(self, net, train_state):
        pass

    def __call__(self, net, train_state=None):
        return None

    def plot(self):
        pass



class List_Train_Monitor(object):

    def __init__(self, train_monitors, default_step=10000):
        self.train_monitors = []
        for t in train_monitors:
            if isinstance(t, tuple):
                t[0].step = t[1]
                self.train_monitors.append(t[0])
            elif (t.step is None):
                t.step = default_step
                self.train_monitors.append(t)
            else:
                self.train_monitors.append(t)

    def names(self):
        return [t.name for t in self.train_monitors]

    def resume(self, net, train_state):
        for t in self.train_monitors:
            t.resume(net, train_state)

    def __call__(self, net, train_state, monitors_data):
        to_print = ""
        for t in self.train_monitors:
            if (train_state["i"] % t.step == 0):
                if isinstance(t, Save_Net):
                    r, p = t(net, train_state, monitors_data)
                else:
                    r, p = t(net, train_state)
                monitors_data[t.name].append((train_state["i"], train_state["time"], r))
                to_print += p
        return monitors_data, to_print



class Running_Loss(Train_Monitor):
    
    def __init__(self, step=None):
        self.name = "running_loss"
        self.step = step
        self.last_loss = 0
        self.last_nbr_examples = 0
        self.round_x_n = lambda x,n : round(x, -int(floor(log10(abs(x)))) + n - 1)

    def resume(self, net, train_state):
        self.last_loss = train_state["total_running_loss"]
        self.last_nbr_examples = train_state["nbr_loss_examples"]

    def __call__(self, net, train_state):
        running_loss = (train_state["total_running_loss"] - self.last_loss)
        running_loss *= 1./(train_state["nbr_loss_examples"] - self.last_nbr_examples)
        self.resume(net, train_state)
        to_print = "loss = {0: <11}".format(self.round_x_n(running_loss, 6))
        return running_loss, to_print

    def plot(self, data, x='i', linecolor='.-b'):
        if data[self.name]:
            plot_error(data[self.name], x=x, linecolor=linecolor)
            plt.ylabel("loss")



class Train_Error(Train_Monitor):
    
    def __init__(self, step=None):
        self.name = "train_error"
        self.step = step
        self.last_error = 0
        self.last_nbr_examples = 0

    def resume(self, net, train_state):
        self.last_error = train_state["total_train_error"]
        self.last_nbr_examples = train_state["nbr_train_examples"]

    def __call__(self, net, train_state):
        train_error = (train_state["total_train_error"] - self.last_error)
        train_error *= 100./(train_state["nbr_train_examples"] - self.last_nbr_examples)
        self.resume(net, train_state)
        to_print = "train_error = {0: <10}".format(round(train_error, 3))
        return train_error, to_print

    def plot(self, data, x='i', linecolor='.-k'):
        if data[self.name]:
            plot_error(data[self.name], x=x, linecolor=linecolor)
    


class Test_Prediction(Train_Monitor):
    
    def __init__(self, testloader, step=None):
        self.name = "test_prediction"
        self.step = step
        self.testloader = testloader
        self.last_i = 0

    def __call__(self, net, train_state=None, recompute=None):
        if (train_state is None):
            if recompute or ((self.last_i is not None) and (recompute is None)):
                self.prediction, self.ground_truth = test_output(net, self.testloader)
                self.last_i = None
        elif (train_state["i"] != self.last_i):
            self.prediction, self.ground_truth = test_output(net, self.testloader)
            self.last_i = train_state["i"]
        return (self.prediction, self.ground_truth), ""



class Test_Error(Train_Monitor):
    
    def __init__(self, testloader_or_test_prediction, step=None):
        self.name = "test_error"
        self.step = step
        self.test_prediction = testloader_or_test_prediction
        if not isinstance(testloader_or_test_prediction, Test_Prediction):
            self.test_prediction = Test_Prediction(testloader_or_test_prediction, step)

    def __call__(self, net, train_state=None, recompute=None):
        res, _ = self.test_prediction(net, train_state, recompute)
        prediction, ground_truth = res
        test_error = prediction_error(prediction, ground_truth, to_print=False)
        to_print = "test_error = {0: <10}".format(round(test_error, 3))
        return test_error, to_print

    def plot(self, data, x='i', linecolor='.-r'):
        if data[self.name]:
            plot_error(data[self.name], x=x, linecolor=linecolor)



class Test_Error_Per_Class(Train_Monitor):
    
    def __init__(self, testloader_or_test_prediction, step=None):
        self.name = "test_error_per_class"
        self.step = step
        self.test_prediction = testloader_or_test_prediction
        if not isinstance(testloader_or_test_prediction, Test_Prediction):
            self.test_prediction = Test_Prediction(testloader_or_test_prediction, step)

    def __call__(self, net, train_state=None, recompute=None):
        res, _ = self.test_prediction(net, train_state, recompute)
        prediction, ground_truth = res
        test_error = prediction_error_per_class(prediction, ground_truth)
        temp = ["{0: <5}".format(round(i, 1)) for i in test_error]
        to_print = "test_error_per_class = {}   ".format(" ".join(temp))
        return test_error, to_print

    def plot(self, data, x='i', linecolor='.-m'):
        if data[self.name]:
            for i in range(len(data[self.name][0][2])):
                d = [(j[0], j[1], j[2][i]) for j in data[self.name]]
                plot_error(d, x=x, linecolor=linecolor)


            
class Test_False_Positive_Per_Class(Train_Monitor):
    
    def __init__(self, testloader_or_test_prediction, step=None):
        self.name = "test_false_positive_per_class"
        self.step = step
        self.test_prediction = testloader_or_test_prediction
        if not isinstance(testloader_or_test_prediction, Test_Prediction):
            self.test_prediction = Test_Prediction(testloader_or_test_prediction, step)

    def __call__(self, net, train_state=None, recompute=None):
        res, _ = self.test_prediction(net, train_state, recompute)
        prediction, ground_truth = res
        test_error = prediction_false_positive_per_class(prediction, ground_truth)
        temp = ["{0: <5}".format(round(i, 1)) for i in test_error]
        to_print = "test_fp_per_class = {}   ".format(" ".join(temp))
        return test_error, to_print

    def plot(self, data, x='i', linecolor='.-m'):
        if data[self.name]:
            for i in range(len(data[self.name][0][2])):
                d = [(j[0], j[1], j[2][i]) for j in data[self.name]]
                plot_error(d, x=x, linecolor=linecolor)



class Save_Net(Train_Monitor):

    def __init__(self, step=None, filename="saved_nets/autosave_net.pth", save_monitors=True):
        self.name = "net_save"
        self.step = step
        self.filename = filename
        self.save_monitors = save_monitors

    def __call__(self, net, train_state={}, monitors_data={}):
        to_save = train_state
        fn = self.filename
        if ("i" in train_state.keys()) and ("{}" in self.filename):
            fn = self.filename.format(i)
        if self.save_monitors:
            to_save = dict(to_save, **monitors_data)
        to_save["net"] = net
        torch.save(to_save, fn)
        return fn, "net saved"

    

def load_net(filename="saved_nets/autosave_net.pth"):
    res = torch.load(filename)
    net = res["net"]
    del res["net"]
    return net, res



class Copy_Net(Train_Monitor):

    def __init__(self, step=None):
        self.name = "net_copy"
        self.step = step
        self.net = None

    def resume(self, net, train_state):
        self.net = copy.deepcopy(net)

    def __call__(self, net, train_state=None):
        self.resume(net, train_state)
        return net, ""
        
            

class Net_Parameters_Change(Train_Monitor):

    def __init__(self, step=None, copy_net=None):
        self.name = "net_parameters_change"
        self.step = step
        self.net = copy_net
        if not isinstance(self.net, Copy_Net):
            self.net = Copy_Net()

    def resume(self, net, train_state):
        self.net.resume(net, train_state)

    def get_weights(self, net, layer_type):
        layers = [(k,j,i) for k,(j,i) in enumerate(net.named_children()) if isinstance(i, layer_type)]
        return [(k,j,i.weight.detach().numpy()) for (k,j,i) in layers]
    
    def __call__(self, net, train_state=None):
        conv_weight = self.get_weights(net, M.conv._ConvNd)
        old_conv_weight = self.get_weights(self.net.net, M.conv._ConvNd)
        conv_diff = [(k,j,np.mean(abs(l-i))) for (n,m,l),(k,j,i) in zip(conv_weight, old_conv_weight)]
        lin_weight = self.get_weights(net, M.linear.Linear)
        old_lin_weight = self.get_weights(self.net.net, M.linear.Linear)
        lin_diff = [(k,j,np.mean(abs(l-i))) for (n,m,l),(k,j,i) in zip(lin_weight, old_lin_weight)]
        res = conv_diff + lin_diff
        res.sort()
        self.resume(net, train_state)
        return res, ""

    def plot(self, data, x='i', linecolor='.-'):
        if data[self.name]:
            leg = []
            for i in range(len(data[self.name][0][2])):
                d = [(j[0], j[1], j[2][i][2]) for j in data[self.name]]
                plot_error(d, x=x, linecolor=linecolor)
                leg.append(str(data[self.name][0][2][i][0]) + " : " + data[self.name][0][2][i][1])
            plt.ylabel("mean abs change")
            plt.legend(leg)



class Show_Conv_Filters(Train_Monitor):

    def __init__(self, step=None):
        self.name = "conv_filters"
        self.step = step

    def get_weights(self, net):
        lay = [(k,j,i) for k,(j,i) in enumerate(net.named_children()) if isinstance(i, M.conv.Conv2d)]
        return [(k,j,i.weight.detach().numpy()) for (k,j,i) in lay]
    
    def __call__(self, net, train_state=None):
        conv_weight = copy.deepcopy(self.get_weights(net))
        return conv_weight, ""

    def plot(self, data, layer=0):
        if data[self.name]:
            leg = []
            try:
                if isinstance(layer, int):
                    ind = [i[0] for i in data[self.name][0][2]].index(layer)
                else:
                    ind = [i[1] for i in data[self.name][0][2]].index(layer)
            except ValueError:
                pass
            ind = 0 # TODO show conv filters for the other layers
            d = np.asarray([j[2][ind][2] for j in data[self.name]])
            mi = d.min()
            ma = d.max()
            d = (d - mi)/(ma - mi)
            filters = [stack_filters(i, axis="y", move_axis=False) for i in d]
            filters = stack_filters(np.stack(filters), axis='x')
            plt.imshow(filters)
            plt.ylabel("filters")
            plt.xlabel("nbr train images (*{})".format(self.step))
