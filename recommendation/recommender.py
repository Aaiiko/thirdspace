import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, Linear
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import torch.nn.functional as F
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class GNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.linear = Linear(hidden_channels, num_features)

    def forward(self, x, edge_index):
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        x = self.linear(x)
        return x

def preprocess_data(user_likes, user_dislikes, all_restaurants, min_stars, feature_weights=None):
    all_data = pd.concat([user_likes, user_dislikes, all_restaurants], ignore_index=True)
    all_data.drop_duplicates(inplace=True)
    all_data.reset_index(drop=True, inplace=True)

    all_data['review'].fillna(3.5, inplace=True)
    all_data = all_data[all_data['review'] >= min_stars]
    all_data['price'].fillna(2, inplace=True)

    # removed price dict substitution
    #price_dict = {'$': 1, '$$': 2, '$$$': 3, '$$$$': 4, 'n/a': 2}
    #all_data['price'] = all_data['price'].map(price_dict)

    # all_data = all_data.dropna(subset=['location'])
    # le_area = LabelEncoder()
    # all_data['Area_encoded'] = le_area.fit_transform(all_data['location'])
    
    category_vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','), lowercase=False, token_pattern=None)
    category_encoded = category_vectorizer.fit_transform(all_data['category'])
    category_df = pd.DataFrame(category_encoded.toarray(), columns=category_vectorizer.get_feature_names_out())
    
    service_vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','), lowercase=False, token_pattern=None)
    service_encoded = service_vectorizer.fit_transform(all_data['tags'])
    service_df = pd.DataFrame(service_encoded.toarray(), columns=service_vectorizer.get_feature_names_out())
    
    scaler = MinMaxScaler()
    all_data['Review_normalized'] = scaler.fit_transform(all_data[['review']])
    
    user_avg_price = user_likes['price'].mean()
    all_data['Price_similarity'] = 1 / (1 + np.abs(all_data['price'] - user_avg_price))
    
    # user_locations = user_likes['location'].unique()
    # all_data['Location_similarity'] = all_data['Area_encoded'].apply(lambda x: 1 if x in user_locations else 0)
    
    feature_df = pd.concat([
        all_data[['Review_normalized', 'Price_similarity']].reset_index(drop=True),
        category_df.reset_index(drop=True),
        service_df.reset_index(drop=True)
    ], axis=1)

    feature_df.drop_duplicates(inplace=True)
    features = feature_df.values
    features_tensor = torch.FloatTensor(features)

    if feature_weights is not None:
        weight_tensor = torch.FloatTensor([
            feature_weights['Star_normalized'],
            feature_weights['Price_normalized'],
            # feature_weights['star_diff_normalized'],
            # feature_weights['Area_encoded'],
            *([1] * category_df.shape[1]),
            *([1] * service_df.shape[1])
        ]).unsqueeze(0)

        features_tensor *= weight_tensor

    return features_tensor, all_data


def create_graph(features, user_likes, user_dislikes):
    num_user_likes = len(user_likes)
    num_user_dislikes = len(user_dislikes)
    num_total = len(features)

    edge_index = []
    edge_types = [] 


    for i in range(num_user_likes):
        for j in range(num_user_likes, num_total):
            edge_index.append([i, j])
            edge_index.append([j, i])
            edge_types.append(1)


    for i in range(num_user_dislikes):
        for j in range(num_user_dislikes, num_total):
            edge_index.append([i + num_user_likes, j]) 
            edge_index.append([j, i + num_user_likes]) 
            edge_types.append(-4) 

    edge_index = torch.LongTensor(edge_index).t().contiguous()
    edge_types = torch.tensor(edge_types, dtype=torch.float).view(-1, 1) 

    return Data(x=features, edge_index=edge_index, edge_attr=edge_types)


def train_model(model, graph_data, lr=0.01, weight_decay=0.001, num_epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(graph_data.x, graph_data.edge_index)
        loss = criterion(out, graph_data.x)
        loss.backward()
        optimizer.step()

def retrain_model(model, graph_data, lr=0.005, weight_decay=0.0005, num_epochs=25):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(graph_data.x, graph_data.edge_index)
        loss = criterion(out, graph_data.x)
        loss.backward()
        optimizer.step()
    
    return model

def get_recommendations(model, graph_data, user_likes, user_dislikes, all_restaurants, top_k=50, temperature=1.0):
    model.eval()
    with torch.no_grad():
        node_embeddings = model(graph_data.x, graph_data.edge_index)

    num_likes = len(user_likes)
    num_dislikes = len(user_dislikes)
    num_user = num_likes + num_dislikes
    user_embeddings = node_embeddings[:num_user]
    restaurant_embeddings = node_embeddings[num_user:]

    if num_likes > 0:
        avg_user_embedding = user_embeddings[:num_likes].mean(dim=0)
    else:
        avg_user_embedding = torch.zeros_like(restaurant_embeddings[0])

    if num_dislikes > 0:
        avg_dislike_embedding = user_embeddings[num_likes:num_user].mean(dim=0)
        avg_user_embedding = avg_user_embedding - 0.1 * avg_dislike_embedding

    similarities = torch.mm(user_embeddings.mean(dim=0).unsqueeze(0), restaurant_embeddings.t()).squeeze()

    scaled_similarities = similarities / temperature

    probabilities = F.softmax(scaled_similarities, dim=0)

    num_samples = min(top_k * 2, len(probabilities))
    indices = torch.multinomial(probabilities, num_samples=num_samples, replacement=False)
    _, indices = torch.sort(similarities, descending=True)

    seen_restaurants = set(user_likes['Name']).union(set(user_dislikes['Name']))
    final_indices = []
    
    for idx in indices:
        restaurant_name = all_restaurants.iloc[idx.item()]['Name']
        if restaurant_name not in seen_restaurants and idx.item() not in final_indices:
            final_indices.append(idx.item())
            if len(final_indices) == top_k:
                break

    return all_restaurants.iloc[final_indices]

def generate_dummy_data():
    np.random.seed(43)  # For reproducibility

    restaurants = pd.DataFrame({
        'Name': [f'Restaurant_{i}' for i in range(15)],
        'review': np.random.uniform(3, 5, 15).round(1),
        'price': np.random.randint(1, 5, 15),
        'category': np.random.choice(['Italian', 'Chinese', 'Mexican', 'American', 'Japanese'], 15),
        'tags': [', '.join(np.random.choice(['Delivery', 'Takeout', 'Dine-in'], np.random.randint(1, 4), replace=False)) for _ in range(15)],
        'location': np.random.choice(['Downtown', 'Suburb', 'Uptown'], 15),
        'Searched City': 'Seattle'
    })

    user_likes = restaurants.sample(n=3)
    user_dislikes = restaurants[~restaurants['Name'].isin(user_likes['Name'])].sample(n=2)
    all_restaurants = restaurants[~restaurants['Name'].isin(user_likes['Name']) & ~restaurants['Name'].isin(user_dislikes['Name'])]
    print(user_likes)
    print(user_dislikes)
    return user_likes, user_dislikes, all_restaurants

def main():
    min_stars = 3.0
    data_file = 'Restaurants_Seattle.csv'
    likes = 'Sample_User.xlsx'
    dislikes = 'User_dislikes.xlsx'

    # user_likes = pd.read_excel(likes)
    # user_dislikes = pd.read_excel(dislikes)
    # data = pd.read_csv(data_file)
    # data = data.drop_duplicates()

    user_likes, user_dislikes, data = generate_dummy_data()

    feature_weights = {
        'Star_normalized': 1.0,
        'Price_normalized': 2.0,
        # 'star_diff_normalized': 1.0,
        # 'Area_encoded': 0.1,
        'Category': 1.0,
        'Services': 1.0
    }
    features, processed_data = preprocess_data(user_likes, user_dislikes, data, min_stars, feature_weights)
    graph_data = create_graph(features, user_likes, user_dislikes)

    model = GNN(num_features=features.shape[1], hidden_channels=64)
    train_model(model, graph_data, lr=0.01, weight_decay=0.001)

    # Adjust top k based on how often we retrain
    recommendations = get_recommendations(model, graph_data, user_likes, user_dislikes, processed_data, top_k=5)
    print(recommendations[['Name', 'review', 'price', 'location', 'category', 'tags', 'Searched City']])

    # READ IN NEW DATA
    # model = retrain_model(model, graph_data, lr=0.01, weight_decay=0.001)
    # recommendations = get_recommendations(model, graph_data, user_likes, user_dislikes, processed_data, top_k=10)
    # print(recommendations[['Name', 'Star', 'Price', 'Area', 'Category', 'Services', 'Searched City']])

if __name__ == "__main__":
    main()