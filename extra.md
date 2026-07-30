docker compose -f compose.yaml -f compose.static.yaml up -d

# To watch logs afterward:

docker compose -f compose.yaml -f compose.static.yaml logs -f deepstream
