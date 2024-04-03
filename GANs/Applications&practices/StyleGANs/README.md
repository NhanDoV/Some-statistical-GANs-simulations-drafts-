## Mở đầu

        Other GANs focused on improving the discriminator in this case we improve the generator. 
        But Style-GAN generates by taking a reference picture.

Style Gan architecture consists of a Mapping network that maps the input to an intermediate Latent space, Further the intermediate is processed using the AdaIN after each layer , there are approximately 18 convolutional layers.

![alt text](https://iq.opengenus.org/content/images/2022/07/8.jpg)

The Style GAN uses the AdaIN or the Adaptive Instance Normalization which is defined as

$$ \text{ADaIN}(x_i, y) = y_{s, i} \dfrac{x_i - \mu(x_i)}{\sigma (x_i)} + y_{b, i} $$


## Kết quả

- Sau 5 epochs