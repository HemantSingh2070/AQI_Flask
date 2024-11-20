@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        city = request.form['city']
        freq = request.form['freq']
        pollutant = request.form['pollutant']

        file_path = f"data/{city}.csv"
        try:
            air_quality_data = pd.read_csv(file_path)
        except FileNotFoundError:
            return render_template(indexHTML, cities=cities, error="File not found for the selected city!")

        air_quality_data.replace(to_replace=-200, value=np.nan, inplace=True)

        # Ensure 'Date' is valid
        air_quality_data['Date'] = pd.to_datetime(air_quality_data['Date'], errors='coerce', dayfirst=True)
        air_quality_data['time'] = "00:00:00"

        # Drop rows with invalid 'Date'
        air_quality_data = air_quality_data.dropna(subset=['Date'])
        air_quality_data['ds'] = air_quality_data['Date'].astype(str) + " " + air_quality_data['time']
        air_quality_data['ds'] = pd.to_datetime(air_quality_data['ds'], errors='coerce')

        # Drop rows with invalid 'ds'
        air_quality_data = air_quality_data.dropna(subset=['ds'])

        # Prepare the dataset for Prophet
        data = pd.DataFrame()
        data['ds'] = air_quality_data['ds']
        data['y'] = pd.to_numeric(air_quality_data[pollutant], errors='coerce')

        # Drop rows with NaN in 'y'
        data = data.dropna(subset=['y'])

        # Validate frequency
        freq_code = freq.split(" ")[0]
        valid_frequencies = ['D', 'W', 'M']
        if freq_code not in valid_frequencies:
            return render_template(indexHTML, cities=cities, error="Invalid frequency selected!")

        # Fit the model
        model = Prophet()
        model.fit(data)

        # Generate future predictions
        future = model.make_future_dataframe(periods=30, freq=freq_code)
        forecast = model.predict(future)

        # Plot the forecast
        fig, ax = plt.subplots()
        model.plot(forecast, ax=ax)

        # Convert plot to base64
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return render_template(indexHTML, cities=cities, plot_url=plot_url, error=None, city=city)

    return render_template(indexHTML, cities=cities, plot_url=None)
