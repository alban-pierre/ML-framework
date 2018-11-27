import numpy as np

from utils.base import *


class Pipeline(Base_Inputs_Block):
    """
    Create a pipeline from several input blocks
    Inputs :
        input_block : Base_Block - [Base_Block] = [] : middle blocks
        first_block : Base_Block - [Base_Block] = [] : setting inputs sets these blocks inputs
        last_block : Base_Block - [Base_Block] = [] : blocks called when you call the pipeline
        **kargs : **{} : Base_Inputs_Block parameters can be set here
            you can also define inside_blocks with kargs by doing set_params(block_name__param=value)
            you can also do set_params(2__param=value) where 2 is the block index
            Note that the index is the one in list (first_block + input_block + last_block)
    Outputs : 
        [self.last_block()] : Note that some blocks can be useless is last_blocks don't call them
    Examples :
        Pipeline([b1, b2, b3], b0, [b4, b5])
        Pipeline([b1(b0), b2(b1), b3(b1)], b0(), [b4(b2), b5(b3)]) # Notation class = instance here
    """
    def __init__(self, input_block=[], first_block=[], last_block=[], **kargs):
        if (not isinstance(input_block, list)):
            input_block = [input_block]
        if (not isinstance(first_block, list)):
            first_block = [first_block]
        if (not isinstance(last_block, list)):
            last_block = [last_block]
        super(Pipeline, self).__init__(first_block + input_block + last_block)
        self.first_block = first_block
        self.last_block = last_block
        if (not self.first_block):
            self.first_block = self.input_block[:1]
        if (not self.last_block):
            self.last_block = self.input_block[-1:]
        self._last_indexes = slice(len(self.input_block) - len(self.last_block), len(self.input_block))
        self.params_names = {"name"}
        for i,b in enumerate(self.input_block):
            self.params_names = self.params_names.union({str(i) + "__" + p for p in b.params_names})
        self.size_input = []
        for b in self.first_block:
            if isinstance(b, Base_Start_Block):
                self.size_input.append(0)
            elif isinstance(b, Base_Input_Block):
                self.size_input.append(1)
            elif isinstance(b, Base_Inputs_Block):
                self.size_input.append(-1)
            elif isinstance(b, Transparent_Block):
                self.size_input.append(1)
            else:
                self.size_input.append(0)
        if (np.sum(np.minimum(self.size_input, 0)) < -1):
            self.size_input = []
        self._init_changed()
        if kargs:
            self.set_params(**kargs)

    def _set_param(self, k, v):
        if ("__" in k):
            block_name, sub_k = k.split("__", 1)
            try:
                i = [s.name for s in self.input_block].index(block_name)
            except ValueError:
                i = int(block_name)
            if isinstance(self.input_block[i], Base_Block):
                self.input_block[i].set_params(**{sub_k:v})
                return False
        elif (k == "input") or (k == "dataset"):
            if self.size_input:
                size_input = copy(self.size_input)
                if (-1 in self.size_input):
                    size_input[size_input.index(-1)] == len(v) - np.sum(size_input) - 1
                cs = 0
                for b,s in zip(self.first_block, size_input):
                    if (cs < len(v)):
                        if (s == 1):
                            b.set_params(input=v[cs])
                            cs += 1
                        elif isinstance(b, Base_Inputs_Block):
                            ns = max(s, 0)
                            b.set_params(input=v[cs:cs+ns])
                            cs += ns
                return False
        elif (k[:5] == "input"):
            le = int(k[6:]) + 1
            if self.size_input:
                size_input = copy(self.size_input)
                if (-1 in self.size_input):
                    size_input[size_input.index(-1)] == le - np.sum(size_input) - 1
                cs = 0
                for b,s in zip(self.first_block, size_input):
                    if (cs < le):
                        if (s == 1) and (cs == le - 1):
                            b.set_params(input=v)
                            cs += 1
                        elif isinstance(b, Base_Inputs_Block):
                            ns = max(s, 0)
                            if (cs + ns >= le):
                                b.set_params(**{"input_"+str(le-1-cs):v})
                            cs += ns
                return False
        return super(Pipeline, self)._set_param(k, v)

    def _get_param(self, k):
        if ("__" in k):
            block_name, sub_k = k.split("__", 1)
            try:
                i = [s.name for s in self.input_block].index(block_name)
            except ValueError:
                i = int(block_name)
            if isinstance(self.input_block[i], Base_Block):
                return self.input_block[i].get_params(sub_k)
        elif (k == "input") or (k == "dataset"):
            return [b.get_params(k) for b in self.first_block if isinstance(b, Base_Block)]
        elif (k[:5] == "input"):
            le = int(k[6:]) + 1
            cs = 0
            for b in self.first_block:
                if isinstance(b, Base_Input_Block) or isinstance(b, Transparent_Block):
                    if cs == le:
                        return b.get_params("input")
                    cs += 1
                elif isinstance(b, Base_Inputs_Block):
                    inp = b.get_params("input")
                    if cs + len(inp) > le:
                        return inp[le - cs]
                    cs += len(inp)
            return []
        return super(Pipeline, self)._get_param(k)

    def changed_train(self, val=None):
        if val is None:
            return super(Pipeline, self).changed_train(self._last_indexes)
        else:
            return super(Pipeline, self).changed_train(val)

    def changed_test(self, val=None):
        if val is None:
            return super(Pipeline, self).changed_test(self._last_indexes)
        else:
            return super(Pipeline, self).changed_test(val)

    def changed_call(self, val=None):
        if val is None:
            return super(Pipeline, self).changed_call(self._last_indexes)
        else:
            return super(Pipeline, self).changed_call(val)

    def changed(self, val=None):
        if val is None:
            return super(Pipeline, self).changed(self._last_indexes)
        else:
            return super(Pipeline, self).changed(val)

    def compute(self):
        if len(self.last_block) == 1:
            return self._input_block_[-1]()
        else:
            return [self._input_block_[i]() for i in self._last_indexes]
            # return self._input_block_[self._last_indexes]()
