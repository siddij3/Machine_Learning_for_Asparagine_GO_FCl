from pandas import DataFrame

def get_dict(tt_feats):
    dict = {
    'Time':tt_feats[:, 0], 
    'Current':tt_feats[:, 1], 
    'Spin Coating':tt_feats[:, 2] ,
    'Increaing PPM':tt_feats[:, 3], 
    'Temperature':tt_feats[:, 4], 
    'Repeat Sensor Use':tt_feats[:, 5] ,
    'Days Elapsed':tt_feats[:, 6],
    'A':tt_feats[:, 7],
    'B':tt_feats[:, 8],
    'C':tt_feats[:, 9],
    'Integrals':tt_feats[:, 10]
    }
    return DataFrame(dict)

def create_dict(str_param, loop_values, R_val, mae_val):

    str_r =  'R {} '.format(str_param)
    str_mae =  'MAE {}'.format(str_param)

    return {
    str_param: [i for i in loop_values],
    str_r    : R_val , 
    str_mae  : mae_val 
    }

def create_dict_two(str_param1, str_param2, loop_values, R_val, mae_val):

    str_r_0 =  f'R - {str_param1} {str_param2} 0'
    str_r_1 =  f'R - {str_param1} {str_param2} 1'

    str_mae_0 =  f'MAE - {str_param1} {str_param2} 0'
    str_mae_1 =  f'MAE - {str_param1} {str_param2} 1'

    return {    
        str_r_0    : [i[0] for i in R_val ], 
        str_r_1    : [i[1] for i in R_val ],     
        str_mae_0  : [i[0] for i in mae_val ], 
        str_mae_1  : [i[1] for i in mae_val ] 
    }

def create_dict_three(str_param1, str_param2, loop_values, R_val, mae_val):

    str_r_0 =  f'R - {str_param1} {str_param2} 0'
    str_r_1 =  f'R - {str_param1} {str_param2} 1'
    str_r_1 =  f'R - {str_param1} {str_param2} 2'

    str_mae_0 =  f'MAE - {str_param1} {str_param2} 0'
    str_mae_1 =  f'MAE - {str_param1} {str_param2} 1'
    str_mae_1 =  f'MAE - {str_param1} {str_param2} 1'

    return {    
        str_r_0    : [i[0] for i in R_val ], 
        str_r_1    : [i[1] for i in R_val ],     
        str_mae_0  : [i[0] for i in mae_val ], 
        str_mae_1  : [i[1] for i in mae_val ] 
    }
