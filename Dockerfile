FROM node:18-alpine

# Çalışma dizinini projenin kökü yapıyoruz
WORKDIR /opt/render/project/src

# Ana dizindeki package.json dosyalarını kopyala ve kur
COPY package*.json ./
RUN npm install --production

# src klasörü, model, config.js ve diğer her şeyi içeri aktar
COPY . .

# Başlatma komutu: chat.js src klasörünün içinde olduğu için tam yolunu veriyoruz
CMD ["node", "src/chat.js"]