mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"kunal.runwal@smail.inf.h-brs.de\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
