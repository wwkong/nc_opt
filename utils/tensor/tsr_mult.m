%{

A special function that 'multiplies' an N1-by-N2-by-...-Nk 'matrix' tensor 
by a 'vector' tensor across the first dimension. 

If type is 'primal' then the 'vector' tensor is N2-by-...-Nk, 
the multiplication is done across the first dimension, and a N1-by-1 
sized vector is generated as output.

If type is 'dual' then the 'vector' tensor is N1-by-1, the
multiplication is done accross the last (k-1) dimensions, and a
N2-by-N3-...-Nk sized tensor is generated as output.

FILE DATA
---------
Last Modified: 
  August 17, 2020
Coders: 
  Weiwei Kong

INPUT
-----
(mat_tsr, vec_tsr):
  The tensors being multiplied.
type:
  The type of multiplication being performed.

OUTPUT
------
out_tsr:
  The output tensor generated by the multiplication.

%}
function out_tsr = tsr_mult(mat_tsr, vec_tsr, type)

  % Reshape
  mat_size = size(mat_tsr);
  vec_size = size(vec_tsr);
  
  % Generates a vector of size N1-by-1.
  if strcmp(type, 'primal')
    if (mat_size(2:end) ~= vec_size)
      error('Dimensions are not compatible!');
    end
    mat_2d_ver = reshape(mat_tsr, [mat_size(1), prod(mat_size(2:end))]);
    vec_1d_ver = reshape(vec_tsr, [prod(vec_size(1:end)), 1]);
    out_tsr = mat_2d_ver * vec_1d_ver;
    
  % Generates a tensor of size N2-by-N3-...-Nk.
  elseif strcmp(type, 'dual')
    if (mat_size(end) ~= vec_size(1))
      error('Dimensions are not compatible!');
    end
    out_size = mat_size(1:(end-1));
    if (length(out_size) == 1)
      out_size = [out_size, 1];
    end
    mat_2d_ver = ...
      reshape(mat_tsr, [prod(out_size), mat_size(end)]);
    out_vec = mat_2d_ver * vec_tsr;
    out_tsr = reshape(out_vec, out_size);
  
  % Improper usage of the function here.
  else
    error('Unknown type! Valid types are: primal, dual.');
  end
  
end