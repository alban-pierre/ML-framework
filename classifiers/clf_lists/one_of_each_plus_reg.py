import os
this_file_path = '/'.join(__file__.split('/')[:-1])
full_path = os.path.join(os.getcwd(), this_file_path)

filename = os.path.join(full_path, "clf_lists/one_of_each.py")
clfs = get_classifiers_from(filename)

from regressions.regressions import get_regressions_from

filename = os.path.join(full_path, "../regressions/reg_lists/one_of_each.py")
regs = get_regressions_from(filename)

classifiers = clfs + [("REG " + i[0], Classifier_From_Regression(i[1])) for i in regs]
