from base import *



class Stack_Output(Base_Inputs_Block):
    """
    Return a list (or tuple etc) of its input blocks
    Inputs :
        input_block : Base_Block - [Base_Block] = [] : input blocks
        *args : *[Base_Block] : You can put the list of inputs as a list of args in __init__
        typ : type = list : the type of the output, if tuple it outputs tuple([input_blocks()])
        **kargs : **{} : Base_Inputs_Block parameters can be set here
    Outputs : 
        typ(self.input_block()) : Note that typ must be an iterator...
    Examples :
        Stack_Output(in_block)
        Stack_Output(in_block, typ=tuple)
    """
    def __init__(self, input_block=[], typ=list, **kargs):
        super(Stack_Output, self).__init__(input_block)
        self.params_names = self.params_names.union({"typ"})
        self.typ = typ
        self.set_params(typ=typ, **kargs)

    def set_params(self, *args, **kargs):
        changed = False
        for k,v in kargs.iteritems():
            changed = self._set_param(k, v) or changed
        if changed:
            self.set_changed_here(True)

    def _set_param(self, k, v):
        if (k == "typ"):
            if (self.typ != v):
                self.typ = v
                return True
            return False
        else:
            return super(Stack_Output, self)._set_param(k, v)

    def _get_param(self, k):
        if (k == "typ"):
            return self.typ
        else:
            return super(Stack_Output, self)._get_param(k)

    def compute(self):
        return self.typ(self._input_block_())

    

# class Concatenate_Output(Base_Inputs_Block):
#     raise NotImplementedError



class Select_Block(Base_Inputs_Block):
    """
    Returns the output of only one of its input blocks, selected with an index
    Inputs :
        input_block : Base_Block - [Base_Block] = [] : input blocks
        *args : *[Base_Block] : You can put the list of inputs as a list of args in __init__
        index : int = 0 : the index of the input block to return
        **kargs : **{} : Base_Inputs_Block parameters can be set here
    Outputs : 
        self.input_block[self.index]() : Note that self.index must be smaller than size of input blocks
    Examples :
        Select_Block(in_block)
        Select_Block(in_block, index=1)
    """
    def __init__(self, input_block=[], index=0, **kargs):
        super(Select_Block, self).__init__(input_block)
        self.params_names = self.params_names.union({"index"})
        self.index = index
        self.set_params(index=index, **kargs)

    # def set_params(self, *args, **kargs):
    #     changed = False
    #     for k,v in kargs.iteritems():
    #         changed = self._set_param(k, v) or changed
    #     if changed:
    #         self.set_changed_here(True)

    def _set_param(self, k, v):
        if (k == "index") or (k == "i"):
            if (self.index != v):
                self.index = v
                return True
            return False
        else:
            return super(Select_Block, self)._set_param(k, v)

    def _get_param(self, k):
        if (k == "index") or (k == "i"):
            return self.index
        else:
            return super(Select_Block, self)._get_param(k)

    def changed_train(self, val=None):
        if val is None:
            return super(Select_Block, self).changed_train(self.index)
        else:
            return super(Select_Block, self).changed_train(val)

    def changed_test(self, val=None):
        if val is None:
            return super(Select_Block, self).changed_test(self.index)
        else:
            return super(Select_Block, self).changed_test(val)

    def changed_call(self, val=None):
        if val is None:
            return super(Select_Block, self).changed_call(self.index)
        else:
            return super(Select_Block, self).changed_call(val)

    def changed(self, val=None):
        if val is None:
            return super(Select_Block, self).changed(self.index)
        else:
            return super(Select_Block, self).changed(val)

    # def __getattr__(self, attr):
    #     if self.__dict__.has_key(attr) or (attr == "index") or (attr == "input_block"):
    #         return self.__dict__[attr]
    #     elif isinstance(self.input_block, Base_Block):
    #         return getattr(self.input_block, attr)
    #     elif (attr[:6] == "output"):
    #         return self.input_block
    #     else:
    #         raise AttributeError("Transparent Block could not find attribute {}".format(attr))

    def compute(self):
        return self._input_block_[self.index]()




class Select_Index(Base_Input_Block):
    """
    Returns some selected lines from the output of its input block
    Inputs :
        input_block : Base_Block = NoBlock : input blocks
        index : int - slice = 0 : the index of the lines of input block to return
        **kargs : **{} : Base_Input_Block parameters can be set here
    Outputs : 
        self.input_block()[self.index]
    Examples :
        Select_Index(in_block)
        Select_Index(in_block, index=1)
        Select_Index(in_block, index=slice(0,3))
        Select_Index(in_block, index=slice(1,-1))
        Select_Index(in_block, index=slice(5,0,-1))
    """
    def __init__(self, input_block=NoBlock, index=0, **kargs):
        super(Select_Index, self).__init__(input_block)
        self.params_names = self.params_names.union({"index"})
        self.index = index
        self.set_params(index=index, **kargs)

    def set_params(self, *args, **kargs):
        changed = False
        for k,v in kargs.iteritems():
            changed = self._set_param(k, v) or changed
        if changed:
            self.set_changed_here(True)

    def _set_param(self, k, v):
        if (k == "index") or (k == "i"):
            if (self.index != v):
                self.index = v
                return True
            return False
        else:
            return super(Select_Index, self)._set_param(k, v)

    def _get_param(self, k):
        if (k == "index") or (k == "i"):
            return self.index
        else:
            return super(Select_Index, self)._get_param(k)

    def compute(self):
        if hasattr(self.index, '__iter__'):
            inp = self._input_block_()
            return type(inp)([inp[i] for i in self.index])
        else:
            return self._input_block_()[self.index]



class Select_Train(Base_Input_Block):
    """
    Same as Select_Index(index=0)
    """
    def compute(self):
        return self._input_block_()[0]



class Select_Test(Base_Input_Block):
    """
    Same as Select_Index(index=1)
    """
    def compute(self):
        return self._input_block_()[1]



class Select_Feature(Base_Input_Block):
    """
    Returns the features of its input, ie [i[0] for i in input_block()]
    Inputs :
        input_block : Base_Block = NoBlock : input blocks
        **kargs : **{} : Base_Input_Block parameters can be set here
    Outputs : 
        self.input_block()[0], and if input_block encodes many datasets, it goes down by one step
    Examples :
        Select_Feature(in_block)
        Select_Feature((x,y))
        Select_Feature([(x1,y1), (x2,y2), (x3,y3)])
    """
    def compute(self):
        inp = self._input_block_()
        if isinstance(inp[0], tuple) or isinstance(inp[0], list):
            return type(inp)([i[0] for i in inp])
        else:
            return inp[0]

        

class Select_Target(Base_Input_Block):
    """
    Returns the targets of its input, ie [i[1] for i in input_block()]
    Inputs :
        input_block : Base_Block = NoBlock : input blocks
        **kargs : **{} : Base_Input_Block parameters can be set here
    Outputs : 
        self.input_block()[1], and if input_block encodes many datasets, it goes down by one step
    Examples :
        Select_Target(in_block)
        Select_Target((x,y))
        Select_Target([(x1,y1), (x2,y2), (x3,y3)])
    """
    def compute(self):
        inp = self._input_block_()
        if isinstance(inp[0], tuple) or isinstance(inp[0], list):
            return type(inp)([i[1] for i in inp])
        else:
            return inp[1]


        
class Train_Test(Base_Input_Block):
    """
    Transform one block into a train_test block : 
        "train" method will return input_block[0]
        "test" method will return input_block[1]
        "()" method will return input_block
    """
    def _train(self):
        return self._input_block()[0]
    
    def _test(self):
        return self._input_block()[1]

    def _call(self):
        return self._input_block()


    
class Redirect_To_Train(Transparent_Block):
    """
    Redirect "test" and "()" to the "train" method of its input block
    """
    def changed_test(self):
        return self.changed_train()

    def changed_call(self):
        return self.changed_train()

    def compute(self):
        self.output = self._input_block.train()
        self.output_train = self.output
        self.output_test = self.output
        self._update_changed()
        return self.output



class Redirect_To_Test(Transparent_Block):
    """
    Redirect "train" and "()" to the "test" method of its input block
    """
    def changed_train(self):
        return self.changed_test()

    def changed_call(self):
        return self.changed_test()

    def compute(self):
        self.output = self._input_block.test()
        self.output_train = self.output
        self.output_test = self.output
        self._update_changed()
        return self.output



class Redirect_To_Call(Base_Input_Block):
    """
    Redirect "train" and "test" to the "()" method of its input block
    """
    def changed_train(self):
        return self.changed_call()

    def changed_test(self):
        return self.changed_call()

    def compute(self):
        self.output = self._input_block()
        self.output_train = self.output
        self.output_test = self.output
        self._update_changed()
        return self.output



class Redirect_Test_To_Train(Transparent_Block):
    """
    Redirect "test" to the "train" method of its input block
    """
    def changed_test(self):
        return self.changed_train()

    def _train(self):
        self.output_test = self._input_block.train()
        self._update_changed_test()
        return self.output_test

    def _test(self):
        self.output_train = self._input_block.train()
        self._update_changed_train()
        return self.output_train

    def _call(self):
        return self._input_block()



class Redirect_Train_To_Test(Transparent_Block):
    """
    Redirect "train" to the "test" method of its input block
    """
    def changed_train(self):
        return self.changed_test()

    def _train(self):
        self.output_test = self._input_block.test()
        self._update_changed_test()
        return self.output_test

    def _test(self):
        self.output_train = self._input_block.test()
        self._update_changed_train()
        return self.output_train

    def _call(self):
        return self._input_block()



class Redirect_Call_To_Train(Transparent_Block):
    """
    Redirect "()" to the "train" method of its input block
    """
    def changed_call(self):
        return self.changed_train()

    def _train(self):
        self.output = self._input_block.train()
        self._update_changed_call()
        return self.output

    def _test(self):
        return self._input_block.test()

    def _call(self):
        self.output_train = self._input_block.train()
        self._update_changed_train()
        return self.output_train



class Redirect_Train_To_Call(Transparent_Block):
    """
    Redirect "train" to the "()" method of its input block
    """
    def changed_train(self):
        return self.changed_call()

    def _train(self):
        self.output = self._input_block()
        self._update_changed_call()
        return self.output

    def _test(self):
        return self._input_block.test()

    def _call(self):
        self.output_train = self._input_block()
        self._update_changed_train()
        return self.output_train



class Redirect_Call_To_Test(Transparent_Block):
    """
    Redirect "()" to the "test" method of its input block
    """
    def changed_call(self):
        return self.changed_test()

    def _train(self):
        return self._input_block.train()

    def _test(self):
        self.output = self._input_block.test()
        self._update_changed_call()
        return self.output

    def _call(self):
        self.output_test = self._input_block.test()
        self._update_changed_test()
        return self.output_test



class Redirect_Test_To_Call(Transparent_Block):
    """
    Redirect "test" to the "()" method of its input block
    """
    def changed_test(self):
        return self.changed_call()

    def _train(self):
        return self._input_block.train()

    def _test(self):
        self.output = self._input_block()
        self._update_changed_call()
        return self.output

    def _call(self):
        self.output_test = self._input_block()
        self._update_changed_test()
        return self.output_test
