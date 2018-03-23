###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch
from torch.autograd import Variable

import data
import pickle

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--vocsize',type=int,default=6000,
		    help='size of vocabulary')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--dictionary', type=str,default="", help="pre-trained dictionary")
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)
model.eval()

if args.cuda:
    model.cuda()
else:
    model.cpu()

if(args.dictionary==""):
	corpus = data.Corpus(args.data,args.vocsize)
else:
	print("WOW")
	corpus = pickle.load(open(args.dictionary,"rb"))	
print("WOW2")
#print(corpus.dictionary.word2idx)

ntokens = len(corpus.dictionary)
hidden = model.init_hidden(1)
input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
if args.cuda:
    input.data = input.data.cuda()

with open(args.outf, 'w') as outf:
    i=0
    while(i<args.words):
        output, hidden = model(input, hidden)
        word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
        
        k=1
        word_idx = torch.multinomial(word_weights, 1)[0]
        while(word_idx==1):
            word_weights[1]=word_weights[1]/1e15
            word_idx = torch.multinomial(word_weights, k)[0]
            k=k+1

        input.data.fill_(word_idx)
        
        try:
            word = corpus.dictionary.idx2word[word_idx]
        except: #KeyError
            word=data.unknownToken
        
        #print(word,word_idx)
        outf.write(word+ ('\n' if i % 20 == 19 else ' '))
        i=i+1

        if i % args.log_interval == 0:
            print('| Generated {}/{} words'.format(i, args.words))
