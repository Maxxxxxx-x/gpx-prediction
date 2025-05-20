import torch
import torch.optim as optim
import torch.nn.functional as F
from environment import Env
from model import LSTMPolicy
import db
import utils
import dotenv
import os

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    dotenv.load_dotenv()
    n_iter = 1000
    learning_rate = 0.0005
    gamma = 0.99
    max_step = 100
    threshold = 10
    delta_t = 5
    for i in range(n_iter):
        # 1. 讀取資料
        connection = db.connect_database()
        if connection is None:
            print("Failed to connect to database")
            return
        record = db.fetch_record_for_training(connection, 500, 1, i)
        if record is None:
            print("Failed to fetch record from database")
            return
        id, data = record[0]
        path = utils.parseGPX(data)
        if path is None:
            continue
        env = Env(path, delta_t=delta_t, threshold=threshold, max_step=max_step)
        
        # 2. 初始化模型、優化器和損失函數
        model = LSTMPolicy(input_size=6, hidden_size=32, hidden_layers=16, output_size=3).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 3. 訓練模型
        for episode in range(100):
            state = env.reset()
            hx = None
            log_probs = []
            rewards = []
            
            for step in range(max_step):
                # 4. 選擇行動
                input_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                logits, hx = model(input_tensor, hx)
                mean = logits.squeeze(0)
                std = torch.ones_like(mean, device=device) * 0.05
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum()
                log_probs.append(log_prob)
                
                # 5. 執行行動
                state, reward, done = env.step(action.detach().cpu().numpy())
                
                # 6. 儲存經驗
                rewards.append(reward)

                if done:
                    break
                
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns, device=device)
            loss = -torch.stack(log_probs) * returns
            loss = loss.sum()

            # 7. 更新模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"ID: {id}: Episode: {episode}, Total Reward: {sum(rewards)}")
        print(f"ID: {id}: Episode: {episode}, Loss: {loss.item()}")
        # 8. 儲存模型
        if os.path.exists("./checkpoint") is False:
            os.makedirs("./checkpoint")
        torch.save(model.state_dict(), f"./checkpoint/model_{i}.pt")
        print(f"Model {i} saved.")

        # 10. 關閉資料庫連線
        connection.close()

    print("Training completed.")

if __name__ == "__main__":
    main()