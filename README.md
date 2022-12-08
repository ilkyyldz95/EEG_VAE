
This code implements the manuscript:
```
I. Yıldız, R. Garner, M. Lai, D. Duncan, “Unsupervised Seizure Identification on EEG”, Computer Methods and Programs in Biomedicine, Vol. 215, 03/2022, pp. 106604.
```

Please use the `pytorch_env.yml` file to install dependencies.

`read_eeg_xxx` files perform preprocessing for each dataset `xxx` out of MIT, UPenn and TUH.

`l1_VAE_xxx` files perform training and testing for each dataset `xxx` out of MIT, UPenn and TUH.

`cluster_raw` file runs clustering competing methods. 
