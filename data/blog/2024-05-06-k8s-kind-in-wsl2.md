---
layout: post # must be 'post'
title: 'Install Kind (Kubernetes in Docker) on WSL2'
subtitle: 'Step-by-Step Guide to Setting Up Kubernetes with Kind on WSL2'
tags:
  - k8s
  - kind
  - docker
  - wsl2
---

# Prerequisite

1.  Get your WSL2 distro ready (I'm using Ubuntu)
2.  Install Docker in WSL2 (I recommend to install Docker Desktop for Windows)
3.  Test if your Docker Desktop is ready

    Ref: https://kind.sigs.k8s.io/docs/user/using-wsl2/#helpful-tips-for-wsl2

    > Notes: Enter `\\wsl$` in windows to view wsl2 files.

# Install Kind and kubectl

0.  If you have internect connection issue, you can forward the request via windows by adding a `.proxy` where I use 1200 in windows for proxy.

```
sorphwer@WINDOWS-C2J5ID6:~$ cat .proxy

#!/bin/bash
hostip=$(cat /etc/resolv.conf |grep -oP '(?<=nameserver\ ).*')
export https_proxy="http://${hostip}:1200"
export http_proxy="http://${hostip}:1200"
export all_proxy="socks5://${hostip}:1200"

sor@WINDOWSXX:~$ source ~/.proxy
sor@WINDOWSXX:~$ wget www.google.com
--2024-05-06 14:47:41--  http://www.google.com/
Connecting to 172.17.144.1:1200... connected.
Proxy request sent, awaiting response... 200 OK
Length: unspecified [text/html]
Saving to: ‘index.html’

index.html                        [ <=>                                              ]  20.46K  --.-KB/s    in 0.07s

2024-05-06 14:47:45 (288 KB/s) - ‘index.html’ saved [20947]

```

1. Install Kind
   ```
   curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.17.0/kind-linux-amd64
   chmod +x ./kind
   sudo mv ./kind /usr/local/bin/kind
   ```
2. Install kubectl

   ```
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
    chmod +x kubectl
    sudo mv kubectl /usr/local/bin
   ```

3. Start Docker (notes: wsl2 does not have systemd, see https://forums.docker.com/t/wsl-cannot-connect-to-the-docker-daemon-at-unix-var-run-docker-sock-is-the-docker-daemon-running/116245)

   ```
   sudo service docker start
   ```

4. Add config
   ```
   # cluster-config.yml
    kind: Cluster
    apiVersion: kind.x-k8s.io/v1alpha4
    nodes:
    - role: control-plane
      extraPortMappings:
      - containerPort: 30000
        hostPort: 30000
        protocol: TCP
   ```
5. Run kind
   ```
    kind create cluster --config=cluster-config.yml
   ```
6. Result
   ```
    sor@WINDOWSXXX:~/k8s$ kind create cluster --config=cluster-config.yml
    Creating cluster "kind" ...
     ✓ Ensuring node image (kindest/node:v1.27.3) 🖼
     ✓ Preparing nodes 📦
     ✓ Writing configuration 📜
     ✓ Starting control-plane 🕹️
     ✓ Installing CNI 🔌
     ✓ Installing StorageClass 💾
    Set kubectl context to "kind-kind"
    You can now use your cluster with:

    kubectl cluster-info --context kind-kind

    Have a nice day! 👋
    sor@WINDOWSXXX:~/k8s$ kubectl cluster-info --context kind-kind
    Kubernetes control plane is running at https://127.0.0.1:36551
    CoreDNS is running at https://127.0.0.1:36551/api/v1/namespaces/kube-system/services/kube-dns:dns/proxy

    To further debug and diagnose cluster problems, use 'kubectl cluster-info dump'.
   ```

Deploy and access:

```
kubectl create deployment nginx --image=nginx --port=80
kubectl create service nodeport nginx --tcp=80:80 --node-port=30000
```

and nginx will be in `localhost:30000`

# Scan k8s and docker image using Trivy

1. Install Trivy

```
wget https://github.com/aquasecurity/trivy/releases/download/v0.18.3/trivy_0.18.3_Linux-64bit.deb
sudo dpkg -i trivy_0.18.3_Linux-64bit.deb

```

2. Add Github token

Trivy require github token to access API , generate one following https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens, and:

```
vim ~/.bash.profile
```

add

```
export GITHUB_TOKEN=github_pat_11AICDMQY0pZxxxxxxxxxxxxxxxxxxxxxxxxx
```

3. Install trivy-plugin-kubectl

```
trivy plugin install github.com/aquasecurity/trivy-plugin-kubectl
```

4. Get pods

```
kubectl get pods
```

5. Scan pods

```
trivy kubectl pod {POD_NAME}
```

For example, we deployed a nginx service in early steps, so we can find the nginx pod and scan it:

```
sor@WINDOWSXXX:~$ kubectl get pods
NAME                    READY   STATUS    RESTARTS   AGE
nginx-55f598f8d-phq4z   1/1     Running   0          39m
sorphwer@WINDOWS-C2J5ID6:~$ trivy kubectl pod nginx-55f598f8d-phq4z
2024-05-06T15:56:58.263+0800    INFO    Detected OS: debian
2024-05-06T15:56:58.263+0800    INFO    Detecting Debian vulnerabilities...
2024-05-06T15:56:58.286+0800    INFO    Number of PL dependency files: 1
2024-05-06T15:56:58.287+0800    INFO    Detecting jar vulnerabilities...

nginx (debian 12.5)
===================
Total: 88 (UNKNOWN: 10, LOW: 9, MEDIUM: 43, HIGH: 23, CRITICAL: 3)
```
