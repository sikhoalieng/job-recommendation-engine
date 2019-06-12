import numpy as np


def get_model(input_shape, word_to_vec_map, word_to_index):
	"""
	Arguments:
	input_shape -- shape of the input, (max_len,)
	word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
	word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

	Returns:
	model -- a model instance in Keras
	"""
	from keras.layers import Input, Flatten, Dense, Dropout, Activation, Concatenate
	from keras.models import Model

	# Input major and job titles
	major_indices = Input(shape=input_shape[0], dtype='int32')
	title_indices = Input(shape=input_shape[1], dtype='int32')
	# Create the embedding layer pretrained with GloVe
	major_embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
	title_embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
	major_embeddings = major_embedding_layer(major_indices)
	title_embeddings = title_embedding_layer(title_indices)
	# Flatten and concatenate pretrained embeddings
	flatten_major = Flatten(name='Pretrained_GloVe_Major_Embedding')(major_embeddings)
	flatten_title = Flatten(name='Pretrained_GloVe_Job_Title_Embedding')(title_embeddings)
	embeddings_merge = Concatenate(name='Major_Job_Title_Embedding')([flatten_major, flatten_title])
	# Modeling interaction between major and job titles
	X3 = Dense(256, activation='relu', name='Fully_Connected_1')(embeddings_merge)

	# Input user features
	degree_input = Input(shape=input_shape[2], name='Degree_Type')
	managed_input = Input(shape=input_shape[3], name='Managed_Others')
	exp_input = Input(shape=input_shape[4], name='Years_of_Experience')
	# Add user features
	X3a = Dense(128, activation='relu', name='Fully_Connected_2')(degree_input)
	X3b = Dense(128, activation='relu', name='Fully_Connected_3')(managed_input)
	X3c = Dense(128, activation='relu', name='Fully_Connected_4')(exp_input)

	categorical_merge = Concatenate(name='Merge_Features')([X3, X3a, X3b, X3c])

	# MLP layer 1
	X4 = Dense(256, activation='relu', name='Fully_Connected_5')(categorical_merge)
	dropout = Dropout(0.5, name='Dropout_2')(X4)

	# MLP layer 2
	X5 = Dense(64, activation='relu', name='Fully_Connected_6')(dropout)
	dropout = Dropout(0.5, name='Dropout_3')(X5)

	# Sigmoid output (probability per training example)
	output = Dense(1, name='Output')(dropout)
	pred = Activation('sigmoid', name='Sigmoid')(output)

	model = Model(inputs=[major_indices, title_indices, degree_input, managed_input, exp_input], outputs=pred)

	return model


def get_train_instances(train, negative_ratio):
	"""
	Produce training instances of positive and negative pairs of (user, job, applied)

	Arguments:
	train -- array of positive instances [userid, jobid]
	negative_ratio -- ratio of negative instances to positive ones

	Returns:
	user_input -- userid
	item_input -- jobid
	labels -- 0 or 1
	"""
	user_input, item_input, labels = [], [], []
	users = np.unique(train[:, 0])
	items = np.unique(train[:, 1])

	index = 0
	for u, i in train:
		temp = []
		# positive instances
		user_input.append(u)
		item_input.append(i)
		labels.append(1)
		index += 1
		temp.append(i)

		# negative instances
		for t in range(len(temp) * negative_ratio):
			j = np.random.randint(items.shape[0])
			while items[j] in temp:
				j = np.random.randint(items.shape[0])
			user_input.append(u)
			item_input.append(items[j])
			labels.append(0)
		if index % 2000 == 0 or index % train.shape[0] == 0:
			print(str(index) + "/" + str(train.shape[0]) + " training instances created")

	return user_input, item_input, labels


def string_to_indices(X, word_to_index, max_len):
	"""
	Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
	The output shape should be such that it can be given to Embedding()

	Arguments:
	X -- array of sentences (strings), of shape (m, 1)
	word_to_index -- a dictionary containing the each word mapped to its index
	max_len -- maximum number of words in a sentence. Assume every sentence in X is no longer than this.

	Returns:
	X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
	"""

	m = X.shape[0]  # number of training examples

	# Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
	X_indices = np.zeros((m, max_len))

	for i in range(m):  # loop over training examples

		# Convert the ith training sentence in lower case and split is into words. You should get a list of words.
		sentence_words = X[i].lower().split()

		# Initialize j to 0
		j = 0

		import re
		# Loop over the words of sentence_words
		for w in sentence_words:
			# Remove all non-alphabetic characters
			regex = re.compile('[^a-z]')
			# First parameter is the replacement, second parameter is your input string
			w = regex.sub('', w)
			try:
				# Set the (i,j)th entry of X_indices to the index of the correct word.
				X_indices[i, j] = word_to_index[w]
			except KeyError:
				# If word not in GloVe dict, skip it (value remains zero)
				pass
			# Increment j to j + 1
			j += 1

	return X_indices


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
	"""
	Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.

	Arguments:
	word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
	word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

	Returns:
	embedding_layer -- pretrained layer Keras instance
	"""
	from keras.layers import Embedding
	vocab_len = len(word_to_index) + 1  # adding 1 to fit Keras embedding requirement
	emb_dim = word_to_vec_map["cucumber"].shape[0]  # define dimensionality of GloVe word vectors (=50)

	# Initialize the embedding matrix as a numpy array of zeros of shape
	# (vocab_len, dimensions of word vectors = emb_dim)
	emb_matrix = np.zeros((vocab_len, emb_dim))

	# Set each row "index" of the embedding matrix to be the word vector representation of the "index"th
	# word of the vocabulary
	for word, index in word_to_index.items():
		emb_matrix[index, :] = word_to_vec_map[word]

	# Define Keras embedding layer with the correct output/input sizes
	embedding_layer = Embedding(vocab_len, 50, trainable=False)

	# Build the embedding layer, it is required before setting the weights of the embedding layer.
	embedding_layer.build((None,))

	# Set the weights of the embedding layer to the embedding matrix. Your layer is now pre-trained.
	embedding_layer.set_weights([emb_matrix])

	return embedding_layer


def convert_to_index(user_input, item_input, train_user, train_item):
	ui = 0
	ii = 0
	user, item = [], []
	print("Begin converting user index")
	for u in user_input:
		user.append(train_user[train_user == u].index)
		ui += 1
		if ui % 20000 == 0 or ui % len(user_input) == 0:
			print(str(ui) + "/" + str(len(user_input)) + " user indices converted")
	print("Begin converting item index")
	for i in item_input:
		item.append(train_item[train_item == i].index)
		ii += 1
		if ii % 20000 == 0 or ii % len(item_input) == 0:
			print(str(ii) + "/" + str(len(item_input)) + " item indices converted")
	return user, item


def get_words_to_index(glove_file):
	with open(glove_file, encoding='utf-8') as f:
		words = set()
		for line in f:
			line = line.strip().split()
			curr_word = line[0]
			words.add(curr_word)

		i = 1
		words_to_index = {}
		for w in sorted(words):
			words_to_index[w] = i
			i = i + 1
	return words_to_index


def strings_to_one_hot(full_set, words):
	from sklearn.preprocessing import LabelBinarizer
	encoder = LabelBinarizer()
	one_hot_fit = encoder.fit(full_set)
	one_hot = one_hot_fit.transform(words)
	return one_hot


def scale_values(full_set, values):
	from sklearn.preprocessing import MinMaxScaler
	full_set = np.array(full_set).reshape(len(full_set), 1)
	values = np.array(values).reshape((len(values), 1))
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled_values_fit = scaler.fit(full_set)
	scaled_values = scaled_values_fit.transform(values)
	return scaled_values
