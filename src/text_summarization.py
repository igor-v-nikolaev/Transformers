from transformers import pipeline

summarizer = pipeline('summarization', device=0)

my_text = """
7.1 Introduction
Age progression, also known as face aging methods, aims to generate
aging or rejuvenating effects on a given face image while maintaining
individual characteristics. In later years, this topic has been the target
of many publications, as it can automatically assist in the search for
missing persons, information forensics, identi ication of criminals, age-
invariant veri ication, and entertainment purposes [39]. In biometric
tasks,1 it allows the reduction of the distance between the individual
features present at the time of training and the current state of the
faces, especially if the training has been done with old images.
Several factors can intensify human aging: solar exposure, smoking,
polluted environment, stress, and genetic factors. Besides, surgical and
non-surgical aesthetic procedures can mitigate the effect of time. In
addition to the intrinsic particularity of the problem, the interferences
caused by other factors (e.g., variations in facial pose, illumination, and
expression) and shortage of labeled aging data make learning face age
progression a relatively complicated problem [36]. Because of these
and other factors, the aging effect is not a deterministic process.
Methods for facial aging can be divided into three categories [29]:
Prototype-based methods: the mean in previously de ined age
groups is estimated, and the differences between the means of thegroups represent the variation in aging between them.
Model-based methods: parametric models are used to simulate the
changes in the elements that make up the face (shape and texture).
Generative methods: generative models are used to generate aged
examples conditioned to an original face.
The natural aging process of a speci ic human must consider some
personalized facial characteristics, e.g., birthmarks, which are almost
invariant with time. Prototype-based age progression methods cannot
preserve individual personality. Once averages are calculated, the
individuality of the examples is attenuated. Consequently, these
methods are not suitable for use in biometric recognition. Model-based
age progression methods require several images of the same individual,
and these over a wide range of ages, dramatically limiting their use in
biometrics. To preserve the personality, [28] proposed a dictionary
learning-based method. A set of age-group-speci ic dictionaries is
learned, and a linear combination of these patterns expresses a unique
personalized aging process.
Many recent publications have utilized generative models for image
generation and have obtained very realistic results for facial aging. It is
possible to generate considerably realistic aged faces with adversarial
autoencoders and generative adversarial networks (GANs). The focus of
this chapter will be the application of generative methods in facial aging
tasks.
In the past, some review papers on age progression were published.
In [29, 37], the authors presented the evolution of facial aging and age
estimation. However, the authors did not mention generative models
since there were no publications on the topic. Finally, in [1], the authors
review multiple GANs applications, including a small section on facial
aging.
The rest of this chapter is organized as follows: Sect. 7.2 presents an
overview of generative adversarial networks beyond reference
architectures and the challenges involved. Section 7.3 provides a list of
available benchmark databases. Section 7.4 details a comparison
between three re-implementations of facial aging methods. Finally,
Sect. 7.5 discusses a conclusion.
"""

summary = summarizer(my_text, max_length=130, min_length=30, do_sample=False)

print(summary[0]['summary_text'])