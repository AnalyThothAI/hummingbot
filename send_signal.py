#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MQTT 交易信号测试工具 - 支持认证

使用示例:
  # 测试连接
  python send_signal.py test

  # 发送买入信号（用 0.1 BNB 买入新币）
  python send_signal.py send --side BUY --base TOKEN_ADDRESS --quote BNB --amount 0.1

  # 发送卖出信号（卖出 1000 个新币）
  python send_signal.py send --side SELL --base TOKEN_ADDRESS --quote BNB --amount 1000

  # 监听信号
  python send_signal.py listen

  # 指定远程服务器
  python send_signal.py send --broker 192.168.1.100 --side BUY --base TOKEN_ADDRESS --amount 0.1

注意：
  - BUY: amount 表示花费的 BNB/WBNB 数量
  - SELL: amount 表示卖出的代币数量
  - BNB 和 WBNB 会自动识别为 WBNB
"""
import json
import time
import argparse
import os
import sys
import paho.mqtt.client as mqtt

# 默认配置
DEFAULT_BROKER = "127.0.0.1"
DEFAULT_PORT = 1883
DEFAULT_TOPIC = "trading/bsc/snipe"

# 从环境变量或 .env 文件读取认证信息
def load_env():
    """加载 .env 文件"""
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

load_env()

MQTT_USER = os.getenv('MQTT_USER', 'hummingbot')
MQTT_PASSWORD = os.getenv('MQTT_PASSWORD', '')

if not MQTT_PASSWORD:
    print("⚠️  警告: 未找到 MQTT_PASSWORD，请检查 .env 文件")

def test_connection(broker, port, username, password):
    """测试 MQTT 连接"""
    print(f"\n测试连接到 {broker}:{port}")
    print(f"用户名: {username}")
    print("=" * 50)
    
    try:
        client = mqtt.Client()
        
        if username and password:
            client.username_pw_set(username, password)
        
        print("正在连接...")
        client.connect(broker, port, 60)
        client.loop_start()
        time.sleep(2)
        
        # 测试发布
        test_topic = "test/connection"
        result = client.publish(test_topic, "test message")
        
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print("✅ 连接成功！")
            print("✅ 发布测试通过")
        else:
            print(f"⚠️  发布失败，错误码: {result.rc}")
        
        client.loop_stop()
        client.disconnect()
        return True
        
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return False

def send_signal(broker, port, topic, username, password,
                side, base, quote, amount, slippage):
    """
    发送交易信号

    计价规则：
    - BUY: amount 表示 quote token (BNB/WBNB) 数量
    - SELL: amount 表示 base token 数量
    """
    print(f"\n发送信号到 {broker}:{port}/{topic}")
    print(f"认证用户: {username}")
    print("=" * 50)

    # 统一处理 BNB/WBNB
    quote_display = quote.upper()
    if quote_display in ["BNB", "WBNB"]:
        quote_display = "BNB/WBNB"

    # 根据交易方向显示说明
    if side.upper() == "BUY":
        print(f"📝 交易说明: 用 {amount} {quote_display} 买入 {base}")
    else:
        print(f"📝 交易说明: 卖出 {amount} 个 {base} 换取 {quote_display}")

    payload = {
        "side": side.upper(),
        "base_token": base,
        "quote_token": quote.upper(),
        "amount": str(amount),
        "slippage": slippage,  # 滑点作为小数 (0.15 = 15%)
        "timestamp": int(time.time())
    }
    
    print("信号内容:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("=" * 50)
    
    try:
        client = mqtt.Client()
        
        if username and password:
            client.username_pw_set(username, password)
        
        print("正在连接...")
        client.connect(broker, port, 60)
        
        msg = json.dumps(payload, ensure_ascii=False)
        result = client.publish(topic, msg)
        
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print("✅ 信号发送成功！")
        else:
            print(f"❌ 发送失败，错误码: {result.rc}")
            
        time.sleep(0.5)
        client.disconnect()
        
    except Exception as e:
        print(f"❌ 发送错误: {e}")
        sys.exit(1)

def listen_signals(broker, port, topic, username, password):
    """监听交易信号"""
    print(f"\n监听信号: {broker}:{port}/{topic}")
    print(f"认证用户: {username}")
    print("按 Ctrl+C 停止监听")
    print("=" * 50)
    
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("✅ 已连接到 MQTT broker")
            client.subscribe(topic)
            client.subscribe("test/#")  # 同时订阅测试主题
            print(f"📡 订阅主题: {topic}")
            print(f"📡 订阅主题: test/#\n")
        else:
            error_messages = {
                1: "连接被拒绝 - 协议版本错误",
                2: "连接被拒绝 - 客户端ID无效",
                3: "连接被拒绝 - 服务器不可用",
                4: "连接被拒绝 - 用户名或密码错误",
                5: "连接被拒绝 - 未授权"
            }
            print(f"❌ 连接失败: {error_messages.get(rc, f'未知错误码 {rc}')}")
            sys.exit(1)
    
    def on_message(client, userdata, msg):
        print(f"\n[{time.strftime('%H:%M:%S')}] 📨 {msg.topic}")
        try:
            payload = json.loads(msg.payload.decode())
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        except:
            print(msg.payload.decode())
        print("-" * 50)
    
    def on_disconnect(client, userdata, rc):
        if rc != 0:
            print(f"\n⚠️  意外断开连接，错误码: {rc}")
    
    try:
        client = mqtt.Client()
        
        if username and password:
            client.username_pw_set(username, password)
        
        client.on_connect = on_connect
        client.on_message = on_message
        client.on_disconnect = on_disconnect
        
        print("正在连接...")
        client.connect(broker, port, 60)
        client.loop_forever()
        
    except KeyboardInterrupt:
        print("\n\n👋 停止监听")
        client.disconnect()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description='MQTT 交易信号工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 测试连接
  %(prog)s test

  # 买入：用 0.1 BNB 买入新币
  %(prog)s send --side BUY --base TOKEN_ADDRESS --quote BNB --amount 0.1

  # 卖出：卖出 1000 个新币
  %(prog)s send --side SELL --base TOKEN_ADDRESS --quote BNB --amount 1000

  # 监听信号
  %(prog)s listen

  # 连接远程服务器
  %(prog)s send --broker 192.168.1.100 --side BUY --base TOKEN_ADDRESS --amount 0.1

说明:
  BUY:  amount 表示花费的 BNB/WBNB 数量
  SELL: amount 表示卖出的代币数量
  BNB 和 WBNB 自动识别为 WBNB
        """
    )
    
    # 通用参数
    parser.add_argument('--broker', default=DEFAULT_BROKER, 
                       help=f'MQTT broker 地址 (默认: {DEFAULT_BROKER})')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, 
                       help=f'MQTT 端口 (默认: {DEFAULT_PORT})')
    parser.add_argument('--username', default=MQTT_USER, 
                       help=f'MQTT 用户名 (默认: {MQTT_USER})')
    parser.add_argument('--password', default=MQTT_PASSWORD, 
                       help='MQTT 密码 (默认: 从 .env 读取)')
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # test 命令
    test_parser = subparsers.add_parser('test', help='测试 MQTT 连接')
    
    # send 命令
    send_parser = subparsers.add_parser('send', help='发送交易信号')
    send_parser.add_argument('--topic', default=DEFAULT_TOPIC, help='MQTT 主题')
    send_parser.add_argument('--side', required=True, choices=['BUY', 'SELL'], 
                           help='交易方向')
    send_parser.add_argument('--base', required=True,
                           help='基础代币 (地址或符号)')
    send_parser.add_argument('--quote', default='BNB',
                           help='计价代币 (默认 BNB，自动识别为 WBNB)')
    send_parser.add_argument('--amount', required=True,
                           help='交易数量 (BUY=BNB数量, SELL=代币数量)')
    send_parser.add_argument('--slippage', type=float, default=0.15,
                           help='滑点容忍度 (小数格式，例如 0.15 = 15%%, 默认 0.15)')
    
    # listen 命令
    listen_parser = subparsers.add_parser('listen', help='监听交易信号')
    listen_parser.add_argument('--topic', default=DEFAULT_TOPIC + '/#', 
                              help='MQTT 主题 (支持通配符)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    if args.command == 'test':
        test_connection(args.broker, args.port, args.username, args.password)
        
    elif args.command == 'send':
        send_signal(
            args.broker, args.port, args.topic,
            args.username, args.password,
            args.side, args.base, args.quote,
            args.amount, args.slippage
        )
        
    elif args.command == 'listen':
        listen_signals(args.broker, args.port, args.topic, 
                      args.username, args.password)

if __name__ == "__main__":
    main()