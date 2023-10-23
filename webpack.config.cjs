const path = require('path');

module.exports = {
    entry: {
        forwardNetwork: './dist/forwardNetwork.js',
    },
    output: {
        libraryTarget: 'umd',
        filename: '[name].bundle.js',
        path: path.resolve(__dirname, 'browser'),
    },
    module: {
        rules: [
            {
                test: /\.js$/,
                exclude: /node_modules/,
                use: {
                loader: 'babel-loader',
                options: {
                    presets: ['@babel/preset-env'],
                },
                },
            },
        ],
    },
};
