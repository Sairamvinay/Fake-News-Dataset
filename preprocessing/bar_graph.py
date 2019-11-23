import matplotlib.pyplot as plt

POS = {'compound': 22698, 'ROOT': 15556, 'punct': 14358, 'nsubj': 8148, 'aux': 3406, 'neg': 588, 'advmod': 2908, 'dobj': 6091, 'mark': 691, 'advcl': 803, 'det': 5584, 'ccomp': 1140, 'nummod': 1779, 'nsubjpass': 310, 'acl': 725, 'prep': 10420, 'amod': 4123, 'pobj': 10502, 'auxpass': 417, 'appos': 2184, 'nmod': 1197, 'cc': 1721, 'conj': 1998, 'intj': 271, 'expl': 36, 'quantmod': 148, 'pcomp': 558, 'relcl': 457, 'xcomp': 866, 'acomp': 561, 'npadvmod': 900, 'attr': 743, 'prt': 486, 'dep': 767, 'poss': 1253, 'case': 452, 'agent': 191, 'csubj': 70, '': 57, 'dative': 87, 'predet': 23, 'oprd': 71, 'preconj': 4, 'meta': 8, 'parataxis': 21, 'csubjpass': 3}

ax = plt.figure().add_subplot(1,1,1)
plt.bar(list(POS.keys()), POS.values(),  color='g')
plt.setp(ax.get_xticklabels(), fontsize=4)
plt.savefig('dep.jpg')
