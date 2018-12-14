import numpy as np
import matplotlib.pyplot as plt
import simplenn

cmap = plt.get_cmap('plasma')

def make_spiral(N, D, K):
    X = np.zeros((N*K, D))
    y = np.zeros(N*K, dtype='uint8')

    for j in range(K):
        ix = range(N*j, N*(j+1))
        r = np.linspace(0, 1, N)
        t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N) * 0.2
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j

    return X, y

def draw_spiral(X, y):
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=cmap)
    plt.show()

def draw_decision_boundary(X, y, K, classify):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, K),
                         np.arange(y_min, y_max, K))
    Z = classify(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=cmap)
    plt.axis('off')
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=cmap)
    plt.show()
    

def main():
    print('Making spiral')
    X, y = make_spiral(100, 2, 3)
    print('Spiral made!')
    draw_spiral(X, y)

    print('Training simplenn against spiral')
    weights = simplenn.train(X, y, 3)
    print('Done!')

    classify = lambda input: np.argmax(simplenn.evaluate(input, *weights), axis=1)
    draw_decision_boundary(X, y, 0.2, classify)

if __name__ == '__main__':
    main()

