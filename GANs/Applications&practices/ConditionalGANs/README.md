 In this GAN the generator and discriminator both are provided with additional information that could be a class label or any modal data. 

 ![alt text](https://iq.opengenus.org/content/images/2022/07/3.jpg)
 
 As the name suggests the additional information helps the discriminator in finding the conditional probability instead of the joint probability.

 $$ \min_G \max_D \mathbb{E}_{x \sim p_r} \log D(x|y) + \mathbb{E}_{z \sim p_z} \log(1 - D(G(z | y))) $$