%{

A special function that 'multiplies' an N1-by-N2-by-...-Nk 'matrix' tensor 
by a 'vector' tensor across the first dimension. 

If type is 'primal' then the 'vector' tensor is N2-by-...-Nk and 
the multiplication is done across the first dimension and a N1-by-1 
sized vector is generated as output.

If type is 'dual' then the 'vector' tensor is NK-by-1 and the
multiplication is done accross the first (k-1) dimensions and a
N1-by-N2-...-N{k-1} sized tensor is generated as output.

%}
function out_vec = nd_multiply(mat_tsr, vec_tsr, type)

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
    out_vec = mat_2d_ver * vec_1d_ver;
    
  % Generates a tensor of size N1-by-N2-...-N{k-1}.
  elseif strcmp(type, 'dual')
  
  % Improper usage of the function here.
  else
    error('Unknown type! Valid types are: primal, dual.');
  end
  
end