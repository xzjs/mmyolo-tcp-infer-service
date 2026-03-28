#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""TCP inference smoke test client.

Usage example:
python deploy/test_tcp_infer_client.py \
  --host 127.0.0.1 \
  --port 9000 \
  --file-path "D:\\piping\\input\\demo.jpg"
"""

import argparse
import json
import socket
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='TCP infer service test client')
    parser.add_argument('--host', default='127.0.0.1', help='server host')
    parser.add_argument('--port', type=int, default=9000, help='server port')
    parser.add_argument('--file-path', required=True, help='file path to infer (can be Windows path)')
    parser.add_argument('--task-id', default='', help='task id, auto-generated when empty')
    parser.add_argument('--config', default='', help='optional config path')
    parser.add_argument('--checkpoint', default='', help='optional checkpoint path')
    parser.add_argument('--device', default='', help='optional device')
    parser.add_argument('--score-thr', type=float, default=None, help='optional score threshold')
    parser.add_argument('--out-dir', default='', help='optional output directory')
    parser.add_argument('--save-visualization', action='store_true', help='save visualization for image task')
    parser.add_argument('--infer-stride', type=int, default=None, help='video infer stride')
    parser.add_argument('--max-frames', type=int, default=None, help='max video frames, -1 means all')
    parser.add_argument('--progress-every-frames', type=int, default=None, help='progress interval for video')
    parser.add_argument('--return-records', action='store_true', help='return detailed records in final response')
    parser.add_argument('--connect-timeout', type=float, default=10.0, help='socket connect timeout in seconds')
    parser.add_argument('--read-timeout', type=float, default=30.0, help='single recv timeout in seconds')
    parser.add_argument('--overall-timeout', type=float, default=7200.0, help='overall task timeout in seconds')
    parser.add_argument('--save-response', default='', help='save final response json to file')
    return parser.parse_args()


def build_request(args: argparse.Namespace) -> Dict[str, Any]:
    task_id = args.task_id.strip() or f'test-{uuid.uuid4().hex[:8]}'
    req: Dict[str, Any] = {
        'task_id': task_id,
        'file_path': args.file_path,
    }

    if args.config:
        req['config'] = args.config
    if args.checkpoint:
        req['checkpoint'] = args.checkpoint
    if args.device:
        req['device'] = args.device
    if args.score_thr is not None:
        req['score_thr'] = args.score_thr
    if args.out_dir:
        req['out_dir'] = args.out_dir
    if args.save_visualization:
        req['save_visualization'] = True
    if args.infer_stride is not None:
        req['infer_stride'] = args.infer_stride
    if args.max_frames is not None:
        req['max_frames'] = args.max_frames
    if args.progress_every_frames is not None:
        req['progress_every_frames'] = args.progress_every_frames
    if args.return_records:
        req['return_records'] = True
    return req


def main() -> int:
    args = parse_args()
    request = build_request(args)

    print(f'[TEST] connect to {args.host}:{args.port}')
    print(f"[TEST] task_id={request['task_id']}")
    print(f"[TEST] file_path={request['file_path']}")

    started = time.time()
    ack_received = False
    final_message: Dict[str, Any] = {}

    with socket.create_connection((args.host, args.port), timeout=args.connect_timeout) as sock:
        sock.settimeout(args.read_timeout)
        payload = json.dumps(request, ensure_ascii=False) + '\n'
        sock.sendall(payload.encode('utf-8'))
        print('[TEST] request sent')

        buffer = b''
        while True:
            elapsed = time.time() - started
            if elapsed > args.overall_timeout:
                print(f'[ERROR] overall timeout ({args.overall_timeout}s) exceeded')
                return 2

            try:
                chunk = sock.recv(4096)
            except socket.timeout:
                print(f'[WAIT] no message in last {args.read_timeout}s, keep waiting...')
                continue

            if not chunk:
                break

            buffer += chunk
            while b'\n' in buffer:
                line, buffer = buffer.split(b'\n', 1)
                if not line:
                    continue
                try:
                    msg = json.loads(line.decode('utf-8'))
                except Exception:
                    print(f'[WARN] non-json message: {line!r}')
                    continue

                msg_type = msg.get('type', '')
                if msg_type == 'ack':
                    ack_received = True
                    print(f"[ACK] task accepted: {msg.get('task_id', '')}")
                    continue

                if msg_type == 'progress':
                    print(
                        f"[PROGRESS] {msg.get('progress', 0)}% "
                        f"stage={msg.get('stage', '')} "
                        f"msg={msg.get('message', '')}"
                    )
                    continue

                if msg_type in {'result', 'error'}:
                    final_message = msg
                    break

            if final_message:
                break

    if not ack_received:
        print('[WARN] ack was not received before finish/close')

    if not final_message:
        print('[ERROR] no final result/error message received')
        return 3

    if final_message.get('type') == 'error':
        print(f"[ERROR] task failed: {final_message.get('message', 'unknown error')}")
        traceback_text = final_message.get('traceback', '')
        if traceback_text:
            print('[ERROR] traceback:')
            print(traceback_text)
        return 1

    result_payload = final_message.get('result', {})
    resolved_input = result_payload.get('resolved_input_path', '')
    task_result = result_payload.get('result', {})
    output_jsonl = task_result.get('output_jsonl', '')
    kind = task_result.get('kind', '')

    print('[OK] task finished')
    print(f'[OK] kind={kind}')
    print(f'[OK] resolved_input_path={resolved_input}')
    print(f'[OK] output_jsonl={output_jsonl}')

    if args.save_response:
        out_path = Path(args.save_response).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(final_message, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        print(f'[OK] final response saved: {out_path}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
