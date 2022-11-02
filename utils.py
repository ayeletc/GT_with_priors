import numpy as np
from scipy.io import savemat, loadmat

def rand_array_fixed_sum(n1,n2, fixed_sum):
    if 'fixed_sum' not in locals():
        fixed_sum = 1
    mat = np.random.rand(n1,n2)#[0]
    return mat * fixed_sum / np.sum(mat)

def split_list_into_2_sequence(list, min_item, max_item):
    a = []
    b = []
    add_to_list = 'a'
    num_of_sequences = 1
    valid_sequenc = True
    for ii, item in enumerate(list[:-1]):
        if add_to_list == 'b':
            b.append(item)
            if list[ii] - list[ii-1] == 1:
                num_of_sequences += 1
                valid_sequenc = False
                break
            
            
        else:
            a.append(item)
            if list[ii+1] - list[ii] != 1:
                add_to_list = 'b'
                num_of_sequences += 1
    if add_to_list == 'a':
        a.append(list[-1])
    else:
        b.append(list[-1])
    if num_of_sequences > 2:
        valid_sequenc = False
    elif num_of_sequences == 2:
        if a[0] != min_item or b[-1] != max_item:
            valid_sequenc = False
    return a, b, valid_sequenc

def save_workspace(filename, names_of_spaces_to_save, dict_of_values_to_save):
    '''
        filename = location to save workspace.
        names_of_spaces_to_save = use dir() from parent to save all variables in previous scope.
            -dir() = return the list of names in the current local scope
        dict_of_values_to_save = use globals() or locals() to save all variables.
            -globals() = Return a dictionary representing the current global symbol table.
            This is always the dictionary of the current module (inside a function or method,
            this is the module where it is defined, not the module from which it is called).
            -locals() = Update and return a dictionary representing the current local symbol table.
            Free variables are returned by locals() when it is called in function blocks, but not in class blocks.

        Example of globals and dir():
            >>> x = 3 #note variable value and name bellow
            >>> globals()
            {'__builtins__': <module '__builtin__' (built-in)>, '__name__': '__main__', 'x': 3, '__doc__': None, '__package__': None}
            >>> dir()
            ['__builtins__', '__doc__', '__name__', '__package__', 'x']
    '''
    print('save_workspace')
    mydic = {}
    for key in names_of_spaces_to_save:
        # print(key, len(key))
        # if len(key) >= 31:
        #     print(key)
        try:
            mydic[key] = dict_of_values_to_save[key]
        except TypeError:
            pass
    
    savemat(filename, mydic, long_field_names=True)



def load_workspace(filename):
    '''
        filename = location to load workspace.
    '''
    mat = loadmat(filename)
    ignore_var = ['__header__', '__version__', '__globals__', 'plot_DD_vs_K_and_T', 'save_workspace', 'load_workspace', 'rand_array_fixed_sum']
    var_dict = {}
    for key in mat.keys():
        if key in ignore_var:
            continue
        if mat[key].dtype in ['<U3','<U8', '<U6', '<U2']:
            str_ar = mat[key]
            var_dict[key] = [string.replace(' ', '') for string in str_ar]
            if key in ['method_DD', 'sampleMethod', 'Tbaseline']:
                var_dict[key] = var_dict[key][0]
            # print(var_dict[key])
        else:
            var_dict[key] = mat[key].squeeze()
    return var_dict
## after getting this var_dict, put it in the globals:
# for key in var_dict.keys():
#     globals()[key] = var_dict[key]

if __name__ == '__main__':
    db_path=r'/Users/ayelet/Library/CloudStorage/OneDrive-Technion/Alejandro/count_possibly_defected_results/shelve_raw/countPDandDD_N20_nmc500_methodDD_Sum_typical_Tbaseline_ML_02082022_224856.mat'
    var_dict = load_workspace(db_path)
    for key in var_dict.keys():
        globals()[key] = var_dict[key]
    pass

    

