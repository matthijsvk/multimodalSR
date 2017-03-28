import sys
import numpy as np

npz_model_file_name = sys.argv[1]
with np.load(npz_model_file_name) as in_file:
	param_values = [in_file['arr_%d' % i] for i in range(len(in_file.files))]


f = open('model_CTC.txt','w')
num_layers = 3
hidden_size = 1824
input_size = 39
output_size = 30
#Number_of_layers
f.write(str(num_layers)+'\n')

#'#1:', 'RecurrentLayer', 'B'
f.write('RB'+'\n')

#'W_in_to_hid' Forward layer
f.write(str(hidden_size)+' '+str(input_size)+'\n')

f.write(' '.join([str(x) for x in param_values[0].reshape(-1)])+'\n' )

#'W_in_to_hid' Backward layer
f.write(str(hidden_size)+' '+str(input_size)+'\n')

f.write(' '.join([ str(x) for x in param_values[3].reshape(-1)])+'\n' )

#'W_hid_to_hid' Forward layer
f.write(str(hidden_size)+' '+str(hidden_size)+'\n')

f.write(' '.join([str(x) for x in param_values[2].reshape(-1)] ) + '\n')

#'W_hid_to_hid' Backward layer
f.write(str(hidden_size)+' '+str(hidden_size)+'\n')

f.write(' '.join([str(x) for x in param_values[5].reshape(-1)] ) +'\n' )


#'b' Forward layer
f.write(str(hidden_size)+'\n')

f.write(' '.join( [str(x) for x in param_values[1].reshape(-1)] ) +'\n')

#'b' Backward layer
f.write(str(hidden_size)+'\n')

f.write(' '.join( [str(x) for x in param_values[4].reshape(-1)] ) +'\n' )

#'#2:', 'DenseLayer', 'U'
f.write('D'+'\n')

#'W',
f.write(str(hidden_size)+' '+str(hidden_size)+'\n')

f.write(' '.join( [str(x) for x in param_values[6].reshape(-1)] ) +'\n' )


#'b',

f.write(str(hidden_size)+'\n')

f.write(' '.join( [str(x) for x in param_values[7].reshape(-1)] ) +'\n' )


#'#2:', 'DenseLayer', 'U'
f.write('D'+'\n')

#'W',
f.write(str(output_size)+' '+str(hidden_size)+'\n')

f.write(' '.join( [str(x) for x in param_values[8].reshape(-1)] ) +'\n')


#'b',

f.write(str(output_size)+'\n')

f.write(' '.join( [str(x) for x in param_values[9].reshape(-1)] )+'\n' )

f.close()