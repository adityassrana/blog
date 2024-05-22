# Updated to latest stable Jekyll 4.2.2 with Ruby 3.1 for better performance and security
# Defines https://hub.docker.com/repository/docker/fastai/fastpages-jekyll
FROM jekyll/jekyll:4.2.2

COPY . .

# Pre-load all gems into the environment with better security practices
RUN chmod -R 755 . && \
    gem install bundler -v 2.4.22 && \
    bundle install --retry=3 --jobs=4 && \
    jekyll build
