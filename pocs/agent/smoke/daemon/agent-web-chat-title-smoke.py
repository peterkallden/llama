#!/usr/bin/env python3
"""Model-free HTTP/JSONL title smoke using a synthetic daemon peer.

Exercises the real web adapter, catalog writes, ordering and adapter restart.
Does not claim model or Cozo verification.
"""
import argparse
import concurrent.futures
import json
import os
import secrets
import socket
import socketserver
import subprocess
import tempfile
import threading
import time
import urllib.request


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--web-binary', required=True)
    args = parser.parse_args()
    resources, plans = {}, {}
    gate, release = threading.Event(), threading.Event()
    token = secrets.token_hex(24)
    lock = threading.Lock()

    class Peer(socketserver.StreamRequestHandler):
        def send(self, value):
            self.wfile.write(json.dumps(value).encode() + b'\n')
            self.wfile.flush()

        def handle(self):
            self.send({'event': 'ready'})
            auth = json.loads(self.rfile.readline())
            require(auth.get('authorization') == token, 'Unexpected daemon authorization')
            self.send({'ok': True})
            command = json.loads(self.rfile.readline())
            kind = command['command']
            if kind == 'run_turn':
                turn = command['turn_id']
                if turn == 'old':
                    gate.set()
                    require(release.wait(15), 'Concurrent turn did not release the old turn')
                if turn == 'no-goal':
                    self.send({'ok': True, 'response': 'Done'})
                    return
                with lock:
                    plans[turn] = {'plan_id': turn, 'goal': 'Goal for ' + turn, 'version': 2}
                self.send({'ok': True, 'response': 'Done', 'plan_id': turn})
                return
            with lock:
                if kind == 'list_resources':
                    result = {'resources': [{'name': value['name'], 'uri': uri} for uri, value in resources.items()]}
                elif kind == 'read_resource':
                    result = {'content': resources[command['uri']]['text']}
                elif kind == 'put_resource':
                    uri = 'resource://metadata/' + str(len(resources))
                    resources[uri] = command
                    result = {'resource': {'uri': uri}}
                elif kind == 'list_plans':
                    require(command['namespace_id'] == 'web' and command['session_id'] == 'web-title-smoke', 'Plan query lost its chat scope')
                    result = {'plans': list(plans.values())}
                elif kind == 'list_sessions':
                    result = {'session_keys': []}
                else:
                    raise RuntimeError('Unexpected command ' + kind)
            self.send({'ok': True, **result})

    class Server(socketserver.ThreadingTCPServer):
        daemon_threads = True
        allow_reuse_address = True

    with Server(('127.0.0.1', 0), Peer) as peer, tempfile.TemporaryFile() as log:
        threading.Thread(target=peer.serve_forever, daemon=True).start()
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        env = dict(os.environ, TITLE_SMOKE_TOKEN=token)

        def start():
            process = subprocess.Popen([args.web_binary, '--listen', '127.0.0.1', '--port', str(port),
                '--daemon-address', '127.0.0.1', '--daemon-port', str(peer.server_address[1]),
                '--daemon-authorization-env', 'TITLE_SMOKE_TOKEN', '--web-bearer-token-env', 'TITLE_SMOKE_TOKEN'],
                env=env, stdout=log, stderr=log)
            for _ in range(100):
                try:
                    with socket.create_connection(('127.0.0.1', port), timeout=.1):
                        return process
                except OSError:
                    time.sleep(.05)
            process.terminate()
            raise RuntimeError('Web adapter did not start')

        def call(path, data=None):
            request = urllib.request.Request('http://127.0.0.1:' + str(port) + path,
                data=None if data is None else json.dumps(data).encode(),
                headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)

        def turn(name, orientation=False):
            return call('/api/v1/turns', {'session_id': 'web-title-smoke', 'project_id': '', 'turn_id': name, 'prompt': 'First question', 'attachment_orientation': orientation})

        process = start()
        try:
            call('/api/v1/chats', {'session_id': 'web-title-smoke'})
            require(turn('first')['chat']['title'] == 'Goal for first', 'First goal was not used')
            require(turn('second')['chat']['title'] == 'Goal for second', 'Existing title was not replaced')
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                older = pool.submit(turn, 'old')
                require(gate.wait(15), 'Old turn did not start')
                require(turn('new')['chat']['title'] == 'Goal for new', 'New turn title was not saved')
                release.set()
                require('chat' not in older.result(), 'Older completion renamed the chat')
            turn('no-goal')
            require('chat' not in turn('attachment-preparation', True), 'Host-only attachment preparation renamed the conversation')
            process.terminate()
            process.wait(timeout=10)
            process = start()
            chat = call('/api/v1/chats')['chats'][0]
            require(chat['title'] == 'Goal for new', 'Title did not survive adapter restart or absent goal')
            print(json.dumps({'ok': True, 'title': chat['title'], 'source': chat['title_source'],
                'checks': ['first-goal', 'rename', 'late-completion', 'missing-goal', 'attachment-preparation', 'adapter-restart'], 'model_used': False}))
        finally:
            release.set()
            process.terminate()
            process.wait(timeout=10)
            peer.shutdown()


if __name__ == '__main__':
    main()
