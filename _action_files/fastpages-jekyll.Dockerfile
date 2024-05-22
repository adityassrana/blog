# Defines https://hub.docker.com/repository/docker/hamelsmu/fastpages-jekyll
FROM jekyll/jekyll:4.1.0
# FROM hamelsmu/fastpages-jekyll

COPY . .

# Pre-load all gems into the environment
RUN chmod -R 777 .
RUN gem install bundler -v 2.4.22
RUN jekyll build
