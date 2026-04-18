import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import math
import jieba
from collections import Counter
import os

# ==================== 1. 读取真实数据集 ====================
data_dir = "./WMT18-English-Chinese-Machine-Translation/train"
en_file = os.path.join(data_dir, "news-commentary-v13.zh-en.en")
zh_file = os.path.join(data_dir, "news-commentary-v13.zh-en.zh")

# 读取所有句子对（如果内存不足，可以使用迭代器分批构建词汇表，这里演示直接读取全部）
print("正在读取数据...")
with open(en_file, 'r', encoding='utf-8') as f_en, open(zh_file, 'r', encoding='utf-8') as f_zh:
    en_lines = [line.strip() for line in f_en.readlines()]
    zh_lines = [line.strip() for line in f_zh.readlines()]

# 确保数量一致
assert len(en_lines) == len(zh_lines), "英文和中文句子数量不一致"
print(f"总句子对数量: {len(en_lines)}")

# 为了演示，可以只使用前 N 条（例如 20000 条）加快训练速度，实际使用时可以全部使用
max_samples = 20000   # 可根据内存/时间调整，全部数据约 23 万条
en_lines = en_lines[:max_samples]
zh_lines = zh_lines[:max_samples]

# ==================== 2. 分词器 ====================
def tokenize_en(text):
    return text.lower().split()

def tokenize_zh(text):
    # 使用 jieba 分词，并过滤掉标点符号（可选）
    words = jieba.cut(text)
    return list(words)

# ==================== 3. 构建词汇表 ====================
def build_vocab_from_sentences(sentences, tokenizer, min_freq=2, max_size=30000):
    counter = Counter()
    for sent in sentences:
        counter.update(tokenizer(sent))
    # 保留最常用的 max_size 个词
    most_common = counter.most_common(max_size - 4)  # 预留4个特殊符号
    vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    for word, _ in most_common:
        if counter[word] >= min_freq:
            vocab[word] = len(vocab)
    return vocab

print("构建英文词汇表...")
src_vocab = build_vocab_from_sentences(en_lines, tokenize_en)
print(f"英文词汇表大小: {len(src_vocab)}")

print("构建中文词汇表...")
tgt_vocab = build_vocab_from_sentences(zh_lines, tokenize_zh)
print(f"中文词汇表大小: {len(tgt_vocab)}")

# ==================== 4. 编码函数和 Dataset ====================
def encode(sentence, vocab, tokenizer, max_len=50):
    tokens = tokenizer(sentence)
    ids = [vocab.get(token, vocab['<unk>']) for token in tokens]
    ids = [vocab['<sos>']] + ids + [vocab['<eos>']]
    if len(ids) > max_len:
        ids = ids[:max_len]
    else:
        ids = ids + [vocab['<pad>'] * (max_len - len(ids))]
    return ids

class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len=50):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src = self.src_sentences[idx]
        tgt = self.tgt_sentences[idx]
        src_ids = encode(src, self.src_vocab, tokenize_en, self.max_len)
        tgt_ids = encode(tgt, self.tgt_vocab, tokenize_zh, self.max_len)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_padded, tgt_padded

# 创建 DataLoader
batch_size = 32   # 根据 GPU 显存调整
dataset = TranslationDataset(en_lines, zh_lines, src_vocab, tgt_vocab, max_len=50)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# ==================== 5. 模型定义（使用您已经修正过的 Transformer 类，这里不再重复）====================
# 请确保您的 Transformer, Position_encoding, MHA, Encoder, Decoder 等类已经在上面的单元格中定义
# 此处假设它们已经定义好了
class Position_encoding(nn.Module):
    def __init__(self,hidden_dim,max_tokens=5000):
        super().__init__()
        position=torch.arange(0,max_tokens,dtype=torch.float).unsqueeze(1) # s h
        pe=torch.zeros(1,max_tokens,hidden_dim) #b s h
        div_terms=torch.exp(torch.arange(0,hidden_dim,2).float()*(-math.log(10000)/hidden_dim))
        pe[0,:,0::2]=torch.sin(position*div_terms)
        pe[0,:,1::2]=torch.cos(position*div_terms)
        self.register_buffer('pe',pe)
    def forward(self,x):
        return x+self.pe[0,:x.size(1),:]
class MHA(nn.Module):
    def __init__(self,hidden_dim,n_heads):
        super().__init__()
        self.wq=nn.Linear(hidden_dim,hidden_dim)
        self.wk=nn.Linear(hidden_dim,hidden_dim)
        self.wv=nn.Linear(hidden_dim,hidden_dim)
        self.wo=nn.Linear(hidden_dim,hidden_dim)
        self.hidden_dim=hidden_dim
        self.n_heads=n_heads
        self.hidden_head=hidden_dim//n_heads
        self.dropout=nn.Dropout(0.1)
    def scaled_dot_product_attention(self,q,k,v,mask=None):
        att=q@k.transpose(-1,-2)/math.sqrt(self.hidden_head)
        if mask is not None:
            att=att.masked_fill(mask==0,float('-inf'))
        att=torch.softmax(att,dim=-1)
        att=self.dropout(att)
        return att@v
    def forward(self,q,k,v,mask=None):
        b = q.size(0)
        q_len = q.size(1)
        k_len = k.size(1)
        q=self.wq(q).view(b,q_len,self.n_heads,-1).transpose(1,2)
        k=self.wk(k).view(b,k_len,self.n_heads,-1).transpose(1,2)
        v=self.wv(v).view(b,k_len,self.n_heads,-1).transpose(1,2)
        score=self.scaled_dot_product_attention(q,k,v,mask)
        score=score.transpose(1,2).contiguous().view(b,q_len,-1)
        return self.wo(score)
class FFN(nn.Module):
    def __init__(self,hidden_dim,dim_ffn,dropout=0.1):
        super().__init__()
        self.linear1=nn.Linear(hidden_dim,dim_ffn)
        self.activation=nn.ReLU()
        self.dropout=nn.Dropout(dropout)
        self.linear2=nn.Linear(dim_ffn,hidden_dim)
    def forward(self,x):
        x=self.linear1(x)
        x=self.activation(x)
        x=self.dropout(x)
        x=self.linear2(x)
        return x
class EncoderLayer(nn.Module):
    def __init__(self,hidden_dim,n_heads,dropout,dim_ffn):
        super().__init__()
        self.attention=MHA(hidden_dim,n_heads)
        self.norm1=nn.LayerNorm(hidden_dim)
        self.dropout1=nn.Dropout(dropout)
        self.ffn=FFN(hidden_dim,dim_ffn,dropout)
        self.norm2=nn.LayerNorm(hidden_dim)
        self.dropout2=nn.Dropout(dropout)
    def forward(self,x,mask=None):
        att=self.attention(x,x,x,mask)
        o1=self.norm1(x+self.dropout1(att))

        ffn=self.ffn(o1)
        return self.norm2(o1+self.dropout2(ffn))
class Encoder(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout, dim_ffn, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(hidden_dim, n_heads, dropout, dim_ffn) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)   # 传递 mask
        x = self.norm(x)
        return x
class DecoderLayer(nn.Module):
    def __init__(self,hidden_dim,n_heads,dropout,dim_ffn):
        super().__init__()
        self.self_attention=MHA(hidden_dim,n_heads)
        self.dropout1=nn.Dropout(dropout)
        self.norm1=nn.LayerNorm(hidden_dim)

        self.cross_attention=MHA(hidden_dim,n_heads)
        self.dropout2=nn.Dropout(dropout)
        self.norm2=nn.LayerNorm(hidden_dim)

        self.ffn=FFN(hidden_dim,dim_ffn,dropout)
        self.norm3=nn.LayerNorm(hidden_dim)
        self.dropout3=nn.Dropout(dropout)
    def forward(self,src,tgt,src_mask=None,tgt_mask=None):
        x=tgt
        att=self.self_attention(x,x,x,tgt_mask)
        o1=self.norm1(x+self.dropout1(att))

        att2=self.cross_attention(o1,src,src,src_mask)
        o2=self.norm2(o1+self.dropout2(att2))

        ffn=self.ffn(o2)
        return self.norm3(o2+self.dropout3(ffn))
class Decoder(nn.Module):
    def __init__(self,hidden_dim,n_heads,dropout,dim_ffn,n_layers):
        super().__init__()
        self.layers=nn.ModuleList([DecoderLayer(hidden_dim,n_heads,dropout,dim_ffn) for i in range(n_layers)])
    def forward(self,src,tgt,src_mask=None,tgt_mask=None):
        x = tgt
        for layer in self.layers:
            x=layer(src,x,src_mask,tgt_mask)
        return x
def create_padding_mask(src, tgt, src_pad_id=0, tgt_pad_id=0):
    src_mask = (src != src_pad_id).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, src_len)
    tgt_mask = (tgt != tgt_pad_id).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, tgt_len)
    tgt_len = tgt.size(1)
    lookahead_mask = torch.ones(tgt_len, tgt_len).tril().bool().unsqueeze(0).unsqueeze(0)
    # 将 lookahead_mask 移到与 tgt_mask 相同的设备
    lookahead_mask = lookahead_mask.to(tgt.device)
    tgt_mask = tgt_mask & lookahead_mask                   # (batch, 1, tgt_len, tgt_len)
    return src_mask, tgt_mask
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, hidden_dim, dim_ffn, n_layers, n_heads, dropout):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, hidden_dim)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, hidden_dim)
        self.pos_encoding = Position_encoding(hidden_dim)   # 添加位置编码
        self.encoder = Encoder(hidden_dim, n_heads, dropout, dim_ffn, n_layers)
        self.decoder = Decoder(hidden_dim, n_heads, dropout, dim_ffn, n_layers)
        self.output_proj = nn.Linear(hidden_dim, tgt_vocab_size)  # 输出层
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt):
        src_mask, tgt_mask = create_padding_mask(src, tgt, src_vocab['<pad>'], tgt_vocab['<pad>'])
        src_emb = self.src_embedding(src) * math.sqrt(self.src_embedding.embedding_dim)
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.tgt_embedding.embedding_dim)
        src_emb = self.pos_encoding(src_emb)   # 添加位置编码
        tgt_emb = self.pos_encoding(tgt_emb)
        src_emb = self.dropout(src_emb)
        tgt_emb = self.dropout(tgt_emb)

        encoder_out = self.encoder(src_emb, src_mask)   # 传递 src_mask
        decoder_out = self.decoder(encoder_out, tgt_emb, src_mask, tgt_mask)  # 传递两个mask
        logits = self.output_proj(decoder_out)   # 映射到词汇表大小
        return logits
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)
hidden_dim = 512          # 可以增大到 512 以获得更好效果
dim_ffn = 2048
n_layers = 6
n_heads = 8
dropout = 0.1
learning_rate = 0.0001
epochs = 20

model = Transformer(src_vocab_size, tgt_vocab_size, hidden_dim, dim_ffn,
                    n_layers, n_heads, dropout).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=src_vocab['<pad>'])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ==================== 6. 训练循环 ====================
print("开始训练真实数据集...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_idx, (src, tgt) in enumerate(dataloader):
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        optimizer.zero_grad()
        output = model(src, tgt_input)            # (batch, tgt_len-1, tgt_vocab_size)
        loss = criterion(output.reshape(-1, tgt_vocab_size), tgt_output.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} finished, Average Loss: {avg_loss:.4f}")

    # 每 2 个 epoch 保存一次模型
    if (epoch + 1) % 2 == 0:
        torch.save(model.state_dict(), f"transformer_epoch_{epoch+1}.pt")
        print(f"Model saved at epoch {epoch+1}")

# ==================== 7. 推理测试 ====================
def translate(model, sentence, src_vocab, tgt_vocab, max_len=50):
    model.eval()
    src_tensor = torch.tensor([encode(sentence, src_vocab, tokenize_en, max_len)]).to(device)
    src_mask = (src_tensor != src_vocab['<pad>']).unsqueeze(1).unsqueeze(2)  # (1,1,1,src_len)

    with torch.no_grad():
        src_emb = model.src_embedding(src_tensor) * math.sqrt(model.src_embedding.embedding_dim)
        src_emb = model.pos_encoding(src_emb)
        src_emb = model.dropout(src_emb)
        encoder_output = model.encoder(src_emb, src_mask)

        tgt_tokens = [tgt_vocab['<sos>']]
        for _ in range(max_len):
            tgt_tensor = torch.tensor([tgt_tokens]).to(device)
            tgt_pad_mask = (tgt_tensor != tgt_vocab['<pad>']).unsqueeze(1).unsqueeze(2)
            tgt_len = tgt_tensor.size(1)
            lookahead_mask = torch.ones(tgt_len, tgt_len).tril().bool().unsqueeze(0).unsqueeze(0).to(device)
            tgt_mask = tgt_pad_mask & lookahead_mask
            tgt_emb = model.tgt_embedding(tgt_tensor) * math.sqrt(model.tgt_embedding.embedding_dim)
            tgt_emb = model.pos_encoding(tgt_emb)
            tgt_emb = model.dropout(tgt_emb)
            decoder_output = model.decoder(encoder_output, tgt_emb, src_mask, tgt_mask)
            logits = model.output_proj(decoder_output)
            next_token = logits[:, -1, :].argmax(dim=-1).item()
            if next_token == tgt_vocab['<eos>']:
                break
            tgt_tokens.append(next_token)

        inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}
        translated = ''.join(inv_tgt_vocab[token] for token in tgt_tokens[1:])
        return translated

# 测试一个句子
test_sentence = "I love machine learning."
print(f"\n输入: {test_sentence}")
print(f"翻译: {translate(model, test_sentence, src_vocab, tgt_vocab)}")