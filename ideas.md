+ understand the accuracy reported
    + These result are calculated using ema model
        + accuracy: accuracy on test set
        + accuracy/valid: accuracy on valid set
        + accuracy/train_labeled: accuracy on labeled train set
    + These "raw" results are calculated with the model without ema
        + accuracy/raw
        + accuracy/raw/train_labeled
        + accuracy/raw/valid
+ what is probe and cta policy?
+ why is default CTAugment?
+ run one experiments, reproduce??
+ think of augmentations to be tried
+ figure out how ctaug select augmentations

# augmentations in CTAugment
+ [x] autocontrast: seems ok and weak
+ [x] blur: quite strong, the parameter from CTAugment seems to assign smaller levels to blur
+ [x] brightness: seems ok and weak
+ color: seems ok and weak
+ contrast: seems ok and weak
+ cutout: not  so strong according to the weight assigned
+ equalize: ok
+ invert: when level=0.5, it becomes gray, so the weight is 0
+ posterize: low level can be bad
+ rescale: large level is bad because it zoomed too much
+ rotate: large angles is broken
+ sharpness: weak
+ shear_x    : weak
+ shear_y    : weak
+ smooth     : weak
+ solarize   : although unnatural, does not have lower weights from CTaug
+ translate_x(y): it was incorrect
    + after fixing the implementation, large translation is zeroed out by CTAugment



## my idea
+ cut mean: seems ok
    + gray seems like a decent mean
    + [x] test it out: slight improvement?
+ rotate_fill: not bad but no significant improvement
    + it seems to allow larger rotate angle
+ since CTAugment can automatically zero out too strong augmentations, 
    + we should be safe in applying augmentations that can be strong? 

+ [x] why is translate_x(y) so weak?
    + there seems to be a mistake in implementation of translate
    + after fixing it, the performance is slightly better 