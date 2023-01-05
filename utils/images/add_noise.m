% SPDX-License-Identifier: MIT
% Copyright © 2022 Weiwei "William" Kong

function noised_im = add_noise(im, snr_in_db)

% Adds 0-mean Gaussian noise with a given SNR (in dB) to an image.
im_normed = double(im - min(im(:))); 
im_normed = im_normed / max(im_normed(:));
noise_var = var(im_normed(:))/10^(snr_in_db/10);
noised_im = imnoise(im_normed, 'gaussian', 0, noise_var);
noised_im = uint8(noised_im * double(max(im(:) - min(im(:)))) + double(min(im(:))));

end