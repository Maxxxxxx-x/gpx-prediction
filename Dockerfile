FROM python:3.13.3-bookworm

WORKDIR /app

RUN apt update && apt upgrade -y

RUN apt install curl gpg -y

RUN curl -fsSL https://pkg.cloudflareclient.com/pubkey.gpg | gpg --yes --dearmor --output /usr/share/keyrings/cloudflare-warp-archive-keyring.gpg

RUN echo "deb [signed-by=/usr/share/keyrings/cloudflare-warp-archive-keyring.gpg] https://pkg.cloudflareclient.com/ bookworm main" | tee /etc/apt/sources.list.d/cloudflare-client.list

RUN apt-get update && apt-get install cloudflare-warp -y

RUN setsid warp-svc >/dev/null 2>&1 < /dev/null &

COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT ["./entrypoint.sh"]
