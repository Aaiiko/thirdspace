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
    all_data['star_diff'] = all_data['Star'] - min_stars
    all_data.drop_duplicates()
    all_data.reset_index()
    price_dict = {'$': 1, '$$': 2, '$$$': 3, '$$$$': 4, 'n/a': 2}
    
    all_data['Price'] = all_data['Price'].map(price_dict)
    all_data['Price'].fillna(2, inplace=True)
    all_data['Star'].fillna(3.5, inplace=True)
    #all_data = all_data.dropna(subset=['Area'])

    # le_area = LabelEncoder()
    # all_data['Area_encoded'] = le_area.fit_transform(all_data['Area'])
    
    category_vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','), lowercase=False, token_pattern=None)
    category_encoded = category_vectorizer.fit_transform(all_data['Category'])
    category_df = pd.DataFrame(category_encoded.toarray(), columns=category_vectorizer.get_feature_names_out())
    category_df.fillna(0, inplace=True)

    service_vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','), lowercase=False, token_pattern=None)
    service_encoded = service_vectorizer.fit_transform(all_data['Services'])
    service_df = pd.DataFrame(service_encoded.toarray(), columns=service_vectorizer.get_feature_names_out())
    service_df.fillna(0, inplace=True)

    scaler = MinMaxScaler()
    all_data[['Star_normalized', 'Price_normalized', 'star_diff_normalized']] = scaler.fit_transform(all_data[['Star', 'Price', 'star_diff']])
    
    all_data['Price_normalized'].fillna(0, inplace=True)
    all_data['Star_normalized'].fillna(0, inplace=True)
    all_data['star_diff_normalized'].fillna(0, inplace=True)

    feature_df = pd.concat([
        all_data[['Star_normalized', 'Price_normalized', 'star_diff_normalized']].reset_index(drop=True),
        category_df.reset_index(drop=True),
        service_df.reset_index(drop=True)
    ], axis=1)

    feature_df.drop_duplicates(inplace=True)
    features = feature_df.values
    features_tensor = torch.FloatTensor(features)
    
    if feature_weights is not None:

        flat_weights = [
            feature_weights.get('Star_normalized', 1.0),
            feature_weights.get('Price_normalized', 1.0),
            feature_weights.get('star_diff_normalized', 1.0),
            feature_weights.get('Category', 1.0),
            feature_weights.get('Services', 1.0)
        ]

        feature_weights_tensor = torch.FloatTensor(flat_weights).unsqueeze(0)

        category_feature_count = category_df.shape[1]

        features_tensor[:, 3:3 + category_feature_count] *= feature_weights_tensor[0, 3]
        
        features_tensor[:, 3 + category_feature_count:] *= feature_weights_tensor[0, 4]

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
            edge_types.append(-1) 

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
        avg_user_embedding = avg_user_embedding - 0.5 * avg_dislike_embedding

    similarities = torch.mm(avg_user_embedding.unsqueeze(0), restaurant_embeddings.t()).squeeze()

    scaled_similarities = similarities / temperature

    probabilities = F.softmax(scaled_similarities, dim=0)

    num_samples = min(top_k * 2, len(probabilities))
    indices = torch.multinomial(probabilities, num_samples=num_samples, replacement=False)

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
    np.random.seed(42)  # For reproducibility

    restaurants = pd.DataFrame({
        'Name': [f'Restaurant_{i}' for i in range(15)],
        'Star': np.random.uniform(3, 5, 15).round(1),
        'Price': np.random.choice(['$', '$$', '$$$', '$$$$'], 15),
        'Category': np.random.choice(['Italian', 'Chinese', 'Mexican', 'American', 'Japanese'], 15),
        'Services': [', '.join(np.random.choice(['Delivery', 'Takeout', 'Dine-in'], np.random.randint(1, 4), replace=False)) for _ in range(15)],
        'Area': np.random.choice(['Downtown', 'Suburb', 'Uptown'], 15),
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
        'Price_normalized': 1.0,
        'star_diff_normalized': 1.0,
        'Category': 1.0,
        'Services': 1.0
    }
    features, processed_data = preprocess_data(user_likes, user_dislikes, data, min_stars, feature_weights)
    graph_data = create_graph(features, user_likes, user_dislikes)

    model = GNN(num_features=features.shape[1], hidden_channels=64)
    train_model(model, graph_data, lr=0.01, weight_decay=0.001)

    # Adjust top k based on how often we retrain
    recommendations = get_recommendations(model, graph_data, user_likes, user_dislikes, processed_data, top_k=5)
    print(recommendations[['Name', 'Star', 'Price', 'Area', 'Category', 'Services', 'Searched City']])

    # READ IN NEW DATA
    # model = retrain_model(model, graph_data, lr=0.01, weight_decay=0.001)
    # recommendations = get_recommendations(model, graph_data, user_likes, user_dislikes, processed_data, top_k=10)
    # print(recommendations[['Name', 'Star', 'Price', 'Area', 'Category', 'Services', 'Searched City']])

if __name__ == "__main__":
    main()