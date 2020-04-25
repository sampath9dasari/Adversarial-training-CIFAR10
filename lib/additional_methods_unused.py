def craft_sample1(x, y, x_gradient):

    x_copy = x.copy()

    _ , acc = imdb_clf.evaluate(x.reshape(-1,maxlen), y.reshape(-1,num_classes), verbose=0)

    for word in range(maxlen):

        if acc<1.0 : break

        word_grad = x_gradient[np.argmax(y), word]
        # print(word_grad.shape)

        jac_sign = np.sign(word_grad).sum()
        vocab_sign = np.add.reduce(np.sign(word_grad - vocab_embeddings),1)
#         jac_sign = np.sign(word_grad)
#         vocab_sign = np.sign(word_grad - vocab_embeddings)

        match_word = np.argmin(np.absolute(vocab_sign - jac_sign))
#         match_word = np.argmin(np.absolute(np.add.reduce(vocab_sign - jac_sign, axis=1)))
        x[word] = match_word

        loss , acc = imdb_clf.evaluate(x.reshape(-1,maxlen), y.reshape(-1,num_classes), verbose=0)
        
#     print(word,acc)
    if acc<1.0:
        return x, word
    else:
        return  x_copy, 0

def craft_sample2(x, y, x_gradient):

    x_copy = x.copy()

    for word in range(maxlen):
        
        pred = np.argmax(imdb_clf.predict_on_batch(x.reshape(-1,maxlen)))
        if pred != y : 
            return x, word

        word_grad = x_gradient[y, word]
        # print(word_grad.shape)

#         jac_sign = np.add.reduce(np.sign(word_grad))
#         vocab_sign = np.add.reduce(np.sign(word_grad - vocab_embeddings),1)
        jac_sign = np.sign(word_grad)
        vocab_sign = np.sign(word_grad - vocab_embeddings)

#         match_word = np.argmin(np.absolute(vocab_sign - jac_sign))
        match_word = np.argmin(np.absolute(np.add.reduce(vocab_sign - jac_sign, axis=1)))
        x[word] = match_word

#         pred = np.argmax(imdb_clf.predict_on_batch(x.reshape(-1,maxlen)))
        
#     print(word,acc)

    return  x_copy, 0