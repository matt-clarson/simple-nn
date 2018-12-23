import sys, pickle
import numpy as np

def get_dimensions(bytes):
    to_int = lambda bytearray: int.from_bytes(bytearray, byteorder='big')
    return [to_int(bytes[i:i+4]) for i in range(4, 16, 4)]

def load_images(bytes):
    num_images, rows, cols = get_dimensions(bytes)
    image_size = rows * cols
    offset = 16
    to_intarray = lambda bytearray: [b for b in bytearray]
    start = lambda i: (i*image_size) + offset
    end = lambda i: ((i+1)*image_size) + offset
    return np.array([to_intarray(bytes[start(i):end(i)])
                        for i in range(num_images)])

def load_labels(bytes):
    return np.array([l for l in bytes[8:]], dtype='uint8')

def save_data(name, nparray):
    with open (f'{name}.pkl', 'wb') as f:
        print(f'Saving to file {name}.pkl')
        pickle.dump(nparray, f)

def process_file(file, pkl_name, labels=False):
    print(f'Loading images from file {file}')
    with open(file, 'rb') as f:
        bytes = f.read()
    nparray = load_labels(bytes) if labels else load_images(bytes)
    save_data(pkl_name, nparray)
    

def main():
    test_images_file = sys.argv[1]
    test_labels_file = sys.argv[2]
    train_images_file = sys.argv[3]
    train_labels_file = sys.argv[4]

    process_file(test_images_file, 'test_images')
    process_file(test_labels_file, 'test_labels', labels=True)
    process_file(train_images_file, 'training_images')
    process_file(train_labels_file, 'training_labels', labels=True)
    
if __name__ == '__main__':
    main()
