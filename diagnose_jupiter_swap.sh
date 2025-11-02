#!/bin/bash

# Jupiter Swap 故障诊断脚本
echo "========================================="
echo "Jupiter Swap 故障诊断"
echo "========================================="
echo ""

# 1. 检查 Gateway 容器状态
echo "1. 检查 Gateway 容器状态:"
docker ps | grep gateway
echo ""

# 2. 检查 Gateway 最近的日志（最后 50 行）
echo "2. Gateway 最近日志（最后 50 行）:"
echo "-----------------------------------"
docker logs --tail 50 gateway
echo ""

# 3. 搜索 InternalServerError 相关日志
echo "3. 搜索 InternalServerError 相关日志:"
echo "-----------------------------------"
docker logs --tail 200 gateway | grep -A 5 -B 5 "InternalServerError\|Error\|error"
echo ""

# 4. 搜索 Jupiter 相关日志
echo "4. 搜索 Jupiter 相关日志:"
echo "-----------------------------------"
docker logs --tail 200 gateway | grep -A 3 -B 3 "jupiter\|Jupiter"
echo ""

# 5. 检查 Gateway 配置
echo "5. Gateway Solana 配置:"
echo "-----------------------------------"
docker exec gateway cat /usr/src/app/conf/solana.yml 2>/dev/null || echo "无法读取配置文件"
echo ""

# 6. 测试 Gateway 健康状态
echo "6. Gateway 健康检查:"
echo "-----------------------------------"
curl -s http://localhost:15888/ | head -20
echo ""

echo "========================================="
echo "诊断完成"
echo "========================================="
echo ""
echo "常见问题："
echo "1. 如果看到 'Transaction simulation failed' - RPC 节点问题或流动性不足"
echo "2. 如果看到 'slippage' - 需要增加 slippage 设置"
echo "3. 如果看到 'insufficient funds' - SOL 余额不足支付 gas"
echo "4. 如果看到 'timeout' - RPC 节点太慢"
echo ""
