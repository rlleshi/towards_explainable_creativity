import gensim

# this script will make the loading of the numberbatch model much faster

PATH = 'resources/numberbatch-en-17.04b.txt'

if __name__ == '__main__':
    print('Loading the model...')

    try:
        numberbatch = gensim.models.KeyedVectors.load_word2vec_format(PATH, binary=False, unicode_errors='ignore')
        print('Converting the model...')
        numberbatch.init_sims(replace=True)
        numberbatch.save('conceptNet')
        print('Convertion successful')
    except FileNotFoundError:
        print('Incorrect file path. Please open the code and modify the PATH variable')
    except ValueError:
        print('The provided file is not a gensim model')