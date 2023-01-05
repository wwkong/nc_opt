% SPDX-License-Identifier: MIT
% Copyright © 2022 Weiwei "William" Kong

function zeroed_im = add_random_zeros(im, density)

% Adds 0s with a given density. A density of 1.0 corresponds to completely
% filling the image with zeros.
[n, m] = size(im);
zeroed_im = reshape(im, [n * m, 1]);
k = ceil(n * m * density);
idx = randsample(n * m, k);
zeroed_im(idx) = 0;
zeroed_im = reshape(zeroed_im, [n, m]);

end