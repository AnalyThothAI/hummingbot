#!/bin/bash

set -e

echo "=========================================="
echo "Hummingbot 一键部署脚本"
echo "=========================================="

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# 检查依赖
echo "检查依赖..."
for cmd in docker docker-compose openssl curl; do
    if ! command -v $cmd &> /dev/null; then
        echo -e "${RED}错误: $cmd 未安装${NC}"
        exit 1
    fi
done
echo -e "${GREEN}✓ 依赖检查完成${NC}"

# 创建目录
echo "创建目录结构..."
mkdir -p redis_data
mkdir -p emqx_data emqx_log
mkdir -p hummingbot_files/{conf,conf/connectors,conf/strategies,conf/scripts,logs,data,scripts}
mkdir -p gateway_files/{conf,logs,db}
mkdir -p shared_certs

echo -e "${GREEN}✓ 目录创建完成${NC}"

# 设置权限
chmod 755 redis_data emqx_data emqx_log hummingbot_files gateway_files
chmod 700 shared_certs

# 生成随机密码
generate_password() {
    openssl rand -base64 16 | tr -d "=+/" | cut -c1-16
}

# 创建 .env 文件
if [ ! -f .env ]; then
    echo "创建 .env 文件..."
    
    MQTT_PASSWORD=$(generate_password)
    GATEWAY_PASS=$(generate_password)
    
    cat > .env << EOF
# ================================
# Hummingbot 环境配置
# ================================

# EMQX Dashboard 登录
EMQX_ADMIN_USER=admin
EMQX_ADMIN_PASSWORD=public

# MQTT 客户端认证
MQTT_USER=hummingbot
MQTT_PASSWORD=${MQTT_PASSWORD}

# Gateway 配置
GATEWAY_PASSPHRASE=${GATEWAY_PASS}
GATEWAY_LOG_LEVEL=info
EOF
    chmod 600 .env
    echo -e "${GREEN}✓ .env 文件创建完成${NC}"
else
    echo -e "${YELLOW}⚠️  .env 文件已存在，跳过创建${NC}"
fi

# 加载环境变量
source .env

# 生成完整的 Gateway 证书链（mTLS）
if [ ! -f shared_certs/ca_cert.pem ]; then
    echo "生成完整的 Gateway mTLS 证书链..."
    cd shared_certs
    
    # 1. 生成 CA (证书颁发机构)
    echo "  1/3 生成 CA 根证书..."
    openssl genrsa -out ca_key.pem 4096 2>/dev/null
    openssl req -new -x509 -key ca_key.pem -out ca_cert.pem -days 365 \
        -subj "/C=US/ST=State/L=City/O=Hummingbot/OU=CA/CN=Hummingbot CA" 2>/dev/null
    
    # 2. 生成 Gateway 服务端证书
    echo "  2/3 生成 Gateway 服务端证书..."
    openssl genrsa -out server_key.pem 4096 2>/dev/null
    openssl req -new -key server_key.pem -out server_csr.pem \
        -subj "/C=US/ST=State/L=City/O=Hummingbot/OU=Gateway/CN=localhost" 2>/dev/null
    openssl x509 -req -in server_csr.pem -CA ca_cert.pem -CAkey ca_key.pem \
        -CAcreateserial -out server_cert.pem -days 365 2>/dev/null
    
    # 3. 生成 Hummingbot 客户端证书
    echo "  3/3 生成 Hummingbot 客户端证书..."
    openssl genrsa -out client_key.pem 4096 2>/dev/null
    openssl req -new -key client_key.pem -out client_csr.pem \
        -subj "/C=US/ST=State/L=City/O=Hummingbot/OU=Client/CN=Hummingbot Client" 2>/dev/null
    openssl x509 -req -in client_csr.pem -CA ca_cert.pem -CAkey ca_key.pem \
        -CAcreateserial -out client_cert.pem -days 365 2>/dev/null
    
    # 4. 清理临时文件
    rm -f server_csr.pem client_csr.pem ca_cert.srl
    
    # 5. 设置权限
    chmod 600 *.pem
    
    cd ..
    echo -e "${GREEN}✓ mTLS 证书链生成完成${NC}"
    echo "  生成的证书:"
    echo "    - CA 根证书: ca_cert.pem, ca_key.pem"
    echo "    - Gateway 服务端: server_cert.pem, server_key.pem"
    echo "    - Hummingbot 客户端: client_cert.pem, client_key.pem"
else
    echo -e "${YELLOW}⚠️  SSL 证书已存在，跳过生成${NC}"
    # 检查是否是完整的证书
    if [ ! -f shared_certs/client_cert.pem ]; then
        echo -e "${RED}⚠️  检测到不完整的证书，建议重新生成${NC}"
        echo "  删除旧证书: rm -rf shared_certs/*.pem"
        echo "  然后重新运行: ./setup.sh"
    fi
fi

# 创建 Gateway 连接配置文件
echo "创建 Hummingbot Gateway 连接配置..."
cat > hummingbot_files/conf/gateway_connections.yml << EOF
gateway:
  gateway_api_host: localhost
  gateway_api_port: 15888
  gateway_api_passphrase: ${GATEWAY_PASSPHRASE}
EOF

chmod 644 hummingbot_files/conf/gateway_connections.yml
echo -e "${GREEN}✓ Gateway 连接配置创建完成${NC}"

# 创建 .gitignore
cat > .gitignore << 'EOF'
# 敏感信息
.env

# 数据目录
redis_data/
emqx_data/
emqx_log/
hummingbot_files/
gateway_files/
shared_certs/

# 日志
*.log

# 临时文件
*.tmp
*.swp
*.bak
*~

# 系统文件
.DS_Store
Thumbs.db
EOF

echo -e "${GREEN}✓ .gitignore 创建完成${NC}"

# 启动 EMQX 并创建用户
echo ""
echo -e "${BLUE}=========================================="
echo "第 1 步：配置 EMQX MQTT Broker"
echo "==========================================${NC}"

docker-compose up -d emqx

echo "等待 EMQX 启动（最多 60 秒）..."
WAIT_TIME=0
MAX_WAIT=60

while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    if docker exec emqx emqx ping 2>/dev/null | grep -q "pong"; then
        echo -e "${GREEN}✓ EMQX 启动成功${NC}"
        break
    fi
    
    echo -n "."
    sleep 2
    WAIT_TIME=$((WAIT_TIME + 2))
done

echo ""

if [ $WAIT_TIME -ge $MAX_WAIT ]; then
    echo -e "${RED}✗ EMQX 启动超时${NC}"
    docker logs emqx --tail 30
    exit 1
fi

# 等待 API 就绪
sleep 5

# 创建 MQTT 用户
echo "创建 MQTT 用户: ${MQTT_USER}"

API_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
  "http://localhost:18083/api/v5/authentication/password_based%3Abuilt_in_database/users" \
  -u "admin:public" \
  -H "Content-Type: application/json" \
  -d "{\"user_id\":\"${MQTT_USER}\",\"password\":\"${MQTT_PASSWORD}\"}" 2>/dev/null)

HTTP_CODE=$(echo "$API_RESPONSE" | tail -n1)

if [ "$HTTP_CODE" = "201" ] || [ "$HTTP_CODE" = "200" ]; then
    echo -e "${GREEN}✓ MQTT 用户创建成功${NC}"
elif [ "$HTTP_CODE" = "409" ]; then
    echo -e "${YELLOW}⚠️  用户已存在${NC}"
else
    echo -e "${YELLOW}⚠️  自动创建失败，稍后可手动创建${NC}"
fi

# 停止单独启动的 EMQX
docker-compose down emqx

# 启动所有服务
echo ""
echo -e "${BLUE}=========================================="
echo "第 2 步：启动所有服务"
echo "==========================================${NC}"

docker-compose up -d

echo -e "${GREEN}✓ 所有服务已启动${NC}"

# 等待 Gateway 启动
echo ""
echo -e "${BLUE}=========================================="
echo "第 3 步：等待 Gateway 初始化"
echo "==========================================${NC}"

echo "等待 Gateway 启动（最多 120 秒）..."
WAIT_TIME=0
MAX_WAIT=120

while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    if curl -k -s https://localhost:15888 2>/dev/null | grep -q "status"; then
        echo ""
        echo -e "${GREEN}✓ Gateway 启动成功${NC}"
        break
    fi
    
    echo -n "."
    sleep 3
    WAIT_TIME=$((WAIT_TIME + 3))
done

echo ""

if [ $WAIT_TIME -ge $MAX_WAIT ]; then
    echo -e "${YELLOW}⚠️  Gateway 启动超时，但可能仍在初始化${NC}"
    echo "可以稍后手动验证: curl -k https://localhost:15888"
else
    # Gateway 启动成功，显示信息
    echo ""
    echo "Gateway API 测试："
    curl -k -s https://localhost:15888
    echo ""
fi

# 测试 MQTT 连接
echo ""
echo -e "${BLUE}=========================================="
echo "第 4 步：验证 MQTT 连接"
echo "==========================================${NC}"

if command -v mosquitto_pub &> /dev/null; then
    echo "发送 MQTT 测试消息..."
    mosquitto_pub -h localhost -p 1883 -t "test/connection" \
        -u "${MQTT_USER}" -P "${MQTT_PASSWORD}" \
        -m "Hello from setup.sh" 2>/dev/null && \
        echo -e "${GREEN}✓ MQTT 连接测试成功${NC}" || \
        echo -e "${YELLOW}⚠️  MQTT 测试失败（请使用 send_signal.py 测试）${NC}"
else
    echo -e "${YELLOW}⚠️  mosquitto-clients 未安装，跳过 MQTT 测试${NC}"
    echo "可以使用 Python 脚本测试: python send_signal.py test"
fi

# 显示服务状态
echo ""
echo -e "${BLUE}=========================================="
echo "服务状态"
echo "==========================================${NC}"

docker-compose ps

# 最终总结
echo ""
echo -e "${GREEN}=========================================="
echo "🎉 部署完成！"
echo "==========================================${NC}"
echo ""
echo -e "${GREEN}配置信息：${NC}"
echo "  📊 MQTT 用户: ${MQTT_USER}"
echo "  🔑 MQTT 密码: ${MQTT_PASSWORD}"
echo "  🔐 Gateway Passphrase: ${GATEWAY_PASSPHRASE}"
echo ""
echo -e "${GREEN}访问地址：${NC}"
echo "  🌐 EMQX Dashboard: http://localhost:18083"
echo "     └─ 登录: admin / public"
echo "  🔗 Gateway API: https://localhost:15888"
echo "  📡 MQTT Broker: localhost:1883"
echo ""
echo -e "${YELLOW}快速测试：${NC}"
echo "  # 测试 Gateway"
echo "  curl -k https://localhost:15888"
echo ""
echo "  # 测试 MQTT（需要先安装依赖: pip install paho-mqtt）"
echo "  python send_signal.py test"
echo ""
echo "  # 进入 Hummingbot"
echo "  docker attach hummingbot"
echo "  # 在 Hummingbot CLI 中执行: gateway status"
echo ""
echo -e "${YELLOW}常用命令：${NC}"
echo "  # 查看日志"
echo "  docker-compose logs -f [service]"
echo ""
echo "  # 重启服务"
echo "  docker-compose restart [service]"
echo ""
echo "  # 停止所有服务"
echo "  docker-compose down"
echo ""
echo -e "${YELLOW}提示：${NC}"
echo "  - Gateway 首次启动可能需要 1-2 分钟完全初始化"
echo "  - 进入 Hummingbot 后使用 'gateway status' 验证连接"
echo "  - 退出 Hummingbot: Ctrl+P, Ctrl+Q（不停止容器）"
echo "  - 所有密码已保存在 .env 文件中"
echo ""
echo -e "${GREEN}🚀 现在可以开始使用 Hummingbot 了！${NC}"
echo ""