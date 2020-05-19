from fitbert import FitBert
fb = FitBert()
masked_string = "Why Bert, you're looking ***mask*** today!"
options = ['buff', 'handsome', 'strong']

ranked_options = fb.rank(masked_string, options=options)
print(ranked_options)
