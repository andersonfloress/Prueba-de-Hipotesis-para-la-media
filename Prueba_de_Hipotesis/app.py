from flask import Flask, render_template, request
import numpy as np
import scipy.stats as stats
import plotly.graph_objs as go
import plotly.io as pio

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test', methods=['POST'])
def test_hypothesis():
    try:
        varianza_conocida = request.form.get('varianza_conocida', 'no')
        media_muestral = float(request.form.get('media_muestral', 0))
        media_poblacional = float(request.form.get('media_poblacional', 0))
        n = int(request.form.get('tamano_muestra', 1))

        if n <= 0:
            raise ValueError("El tamaño de la muestra debe ser mayor que cero")
        
        alpha = float(request.form.get('alpha', 0.05))
        hipotesis_alternativa = request.form.get('hipotesis_alternativa', 'igual')

        acceptance_area = None  # Inicializa acceptance_area
        rejection_area_izquierda = None  # Inicializa rejection_area_izquierda
        rejection_area_derecha = None  # Inicializa rejection_area_derecha

        if varianza_conocida == 'si':
            desviacion_estandar_poblacion = float(request.form.get('desviacion_estandar_poblacion', 0))
            if desviacion_estandar_poblacion <= 0:
                raise ValueError("La desviación estándar poblacional debe ser mayor que cero")
            z = (media_muestral - media_poblacional) / (desviacion_estandar_poblacion / np.sqrt(n))
            z = round(z, 2)
            if hipotesis_alternativa == 'diferente':  # Para pruebas bilaterales
                z_alpha = abs(stats.norm.ppf(alpha / 2))  # Valor crítico bilateral
                z_alpha = round(z_alpha, 2)
            else:
                z_alpha = stats.norm.ppf(1 - alpha) if hipotesis_alternativa in ['mayor', 'mayor_igual'] else stats.norm.ppf(alpha)
                z_alpha = round(z_alpha, 2)
        else:
            desviacion_estandar_muestra = float(request.form.get('desviacion_estandar_muestra', 0))
            if desviacion_estandar_muestra <= 0:
                raise ValueError("La desviación estándar muestral debe ser mayor que cero")
            t = (media_muestral - media_poblacional) / (desviacion_estandar_muestra / np.sqrt(n))
            t = round(t, 2)
            df = n - 1
            t_alpha = stats.t.ppf(1 - alpha, df) if hipotesis_alternativa in ['mayor', 'mayor_igual'] else stats.t.ppf(alpha, df)
            t_alpha = round(t_alpha, 2)

        # Decisión basada en el valor crítico
        if varianza_conocida == 'si':
            if hipotesis_alternativa == 'mayor':
                resultado = "Se Rechaza la H0" if z > z_alpha else "NO Se rechaza la H0"
            elif hipotesis_alternativa == 'menor':
                resultado = "Se Rechaza la H0" if z < z_alpha else "NO Se rechaza la H0"
            elif hipotesis_alternativa == 'diferente':
                z_alpha_dos_colas = abs(stats.norm.ppf(alpha / 2))
                resultado = "Se Rechaza la H0" if abs(z) > z_alpha_dos_colas else "NO Se rechaza la H0"
            else:
                resultado = "NO Se rechaza la H0"  # Por defecto, no rechazar
        else:
            if hipotesis_alternativa == 'mayor':
                resultado = "Se Rechaza la H0" if t > t_alpha else "NO Se rechaza la H0"
            elif hipotesis_alternativa == 'menor':
                resultado = "Se Rechaza la H0" if t < t_alpha else "NO Se rechaza la H0"
            elif hipotesis_alternativa == 'diferente':
                t_alpha_dos_colas = abs(stats.t.ppf(alpha / 2, df))
                resultado = "Se Rechaza la H0" if abs(t) > t_alpha_dos_colas else "NO Se rechaza la H0"
            else:
                resultado = "NO Se rechaza la H0"  # Por defecto, no rechazar

        # Crear el gráfico
        if varianza_conocida == 'si':
            x = np.linspace(-4, 4, 100)
            y = stats.norm.pdf(x)
            trace = go.Scatter(x=x, y=y, mode='lines', name='Distribución', hoverinfo='name')
            critical_point = go.Scatter(x=[z_alpha], y=[stats.norm.pdf(z_alpha)], mode='markers', marker=dict(color='red', size=10), name='Punto Crítico')
            test_statistic_point = go.Scatter(x=[z], y=[stats.norm.pdf(z)], mode='markers', marker=dict(color='blue', size=10), name='Valor de Z')
        else:
            x = np.linspace(-4, 4, 100)
            y = stats.t.pdf(x, df)
            trace = go.Scatter(x=x, y=y, mode='lines', name='Distribución', hoverinfo='name')
            critical_point = go.Scatter(x=[t_alpha], y=[stats.t.pdf(t_alpha, df)], mode='markers', marker=dict(color='red', size=10), name='Punto Crítico')
            test_statistic_point = go.Scatter(x=[t], y=[stats.t.pdf(t, df)], mode='markers', marker=dict(color='blue', size=10), name='Valor de T')

        # Definir el layout
        layout = go.Layout(
            title='Grafico de Regiones de Aceptación y Rechazo',
            xaxis=dict(title='Valores'),
            yaxis=dict(title='Densidad de Probabilidad'),
            showlegend=True,
            paper_bgcolor='rgba(240,240,240,1)'
        )

        # Crear la región de aceptación y rechazo
        if hipotesis_alternativa == 'diferente':
            # Para la prueba bilateral, agregar puntos críticos en ambos lados
            if varianza_conocida == 'si':
                z_alpha_izquierda = -abs(stats.norm.ppf(alpha / 2))  # Punto crítico izquierdo
                z_alpha_derecha = abs(stats.norm.ppf(alpha / 2))  # Punto crítico derecho
                rejection_area_x_izquierda = np.linspace(-4, z_alpha_izquierda, 100)
                rejection_area_x_derecha = np.linspace(z_alpha_derecha, 4, 100)
                acceptance_area_x = np.linspace(z_alpha_izquierda, z_alpha_derecha, 100)
                rejection_area_y_izquierda = stats.norm.pdf(rejection_area_x_izquierda)
                rejection_area_y_derecha = stats.norm.pdf(rejection_area_x_derecha)
                acceptance_area_y = stats.norm.pdf(acceptance_area_x)
                
                #Agregar puntos criticosen izquierda y derecha
                critical_point_izquierda = go.Scatter(x=[z_alpha_izquierda], y=[stats.norm.pdf(z_alpha_izquierda)], mode='markers', marker=dict(color='red', size=10), name='Punto Crítico Izquierda')
                critical_point_derecha = go.Scatter(x=[z_alpha_derecha], y=[stats.norm.pdf(z_alpha_derecha)], mode='markers', marker=dict(color='red', size=10), name='Punto Crítico Derecha')
                
                #Definir áreas de aceptación y rechazo
                acceptance_area = go.Scatter(x=acceptance_area_x, y=acceptance_area_y, fill='tozeroy', fillcolor='rgba(0,255,0,0.5)', mode='none', name='Región de Aceptación', hoverinfo='name')
                rejection_area_izquierda = go.Scatter(x=rejection_area_x_izquierda, y=rejection_area_y_izquierda, fill='tozeroy', fillcolor='rgba(255,0,0,0.5)', mode='none', name='Región de Rechazo Izquierda', hoverinfo='name')
                rejection_area_derecha = go.Scatter(x=rejection_area_x_derecha, y=rejection_area_y_derecha, fill='tozeroy', fillcolor='rgba(255,0,0,0.5)', mode='none', name='Región de Rechazo Derecha')

                fig = go.Figure(data=[trace, acceptance_area, rejection_area_izquierda, rejection_area_derecha, critical_point_izquierda, critical_point_derecha, test_statistic_point], layout=layout)
            else:
                t_alpha_izquierda = -abs(stats.t.ppf(alpha / 2, df))  # Punto crítico izquierdo
                t_alpha_derecha = abs(stats.t.ppf(alpha / 2, df))  # Punto crítico derecho
                rejection_area_x_izquierda = np.linspace(-4, t_alpha_izquierda, 100)
                rejection_area_x_derecha = np.linspace(t_alpha_derecha, 4, 100)
                acceptance_area_x = np.linspace(t_alpha_izquierda, t_alpha_derecha, 100)
                rejection_area_y_izquierda = stats.t.pdf(rejection_area_x_izquierda, df)
                rejection_area_y_derecha = stats.t.pdf(rejection_area_x_derecha, df)
                acceptance_area_y = stats.t.pdf(acceptance_area_x, df)

                critical_point_izquierda = go.Scatter(x=[t_alpha_izquierda], y=[stats.t.pdf(t_alpha_izquierda, df)], mode='markers', marker=dict(color='red', size=10), name='Punto Crítico Izquierda')
                critical_point_derecha = go.Scatter(x=[t_alpha_derecha], y=[stats.t.pdf(t_alpha_derecha, df)], mode='markers', marker=dict(color='red', size=10), name='Punto Crítico Derecha')

                acceptance_area = go.Scatter(x=acceptance_area_x, y=acceptance_area_y, fill='tozeroy', fillcolor='rgba(0,255,0,0.5)', mode='none', name='Región de Aceptación', hoverinfo='name')
                rejection_area_izquierda = go.Scatter(x=rejection_area_x_izquierda, y=rejection_area_y_izquierda, fill='tozeroy', fillcolor='rgba(255,0,0,0.5)', mode='none', name='Región de Rechazo Izquierda', hoverinfo='name')
                rejection_area_derecha = go.Scatter(x=rejection_area_x_derecha, y=rejection_area_y_derecha, fill='tozeroy', fillcolor='rgba(255,0,0,0.5)', mode='none', name='Región de Rechazo Derecha')

                fig = go.Figure(data=[trace, acceptance_area, rejection_area_izquierda, rejection_area_derecha, critical_point_izquierda, critical_point_derecha, test_statistic_point], layout=layout)
        else:
            # Prueba unilateral
            if varianza_conocida == 'si':
                if hipotesis_alternativa == 'menor':
                    rejection_area_x = np.linspace(-4, z_alpha, 100)
                    acceptance_area_x = np.linspace(z_alpha, 4, 100)
                else:
                    rejection_area_x = np.linspace(z_alpha, 4, 100)
                    acceptance_area_x = np.linspace(-4, z_alpha, 100)
                rejection_area_y = stats.norm.pdf(rejection_area_x)
                acceptance_area_y = stats.norm.pdf(acceptance_area_x)
                
                #Definir las áreas
                acceptance_area = go.Scatter(x=acceptance_area_x, y=acceptance_area_y, fill='tozeroy', fillcolor='rgba(0,255,0,0.5)', mode='none', name='Región de Aceptación', hoverinfo='name')
                rejection_area = go.Scatter(x=rejection_area_x, y=rejection_area_y, fill='tozeroy', fillcolor='rgba(255,0,0,0.5)', mode='none', name='Región de Rechazo', hoverinfo='name')
            else:
                if hipotesis_alternativa == 'menor':
                    rejection_area_x = np.linspace(-4, t_alpha, 100)
                    acceptance_area_x = np.linspace(t_alpha, 4, 100)
                else:
                    rejection_area_x = np.linspace(t_alpha, 4, 100)
                    acceptance_area_x = np.linspace(-4, t_alpha, 100)
                rejection_area_y = stats.t.pdf(rejection_area_x, df)
                acceptance_area_y = stats.t.pdf(acceptance_area_x, df)
                
                #Definir las áreas
                acceptance_area = go.Scatter(x=acceptance_area_x, y=acceptance_area_y, fill='tozeroy', fillcolor='rgba(0,255,0,0.5)', mode='none', name='Región de Aceptación', hoverinfo='name')
                rejection_area = go.Scatter(x=rejection_area_x, y=rejection_area_y, fill='tozeroy', fillcolor='rgba(255,0,0,0.5)', mode='none', name='Región de Rechazo', hoverinfo='name')

        # Crear áreas de aceptación y rechazo para pruebas bilaterales
        if hipotesis_alternativa == 'diferente':
            fig = go.Figure(data=[trace, acceptance_area, rejection_area_izquierda, rejection_area_derecha, critical_point_izquierda, critical_point_derecha, test_statistic_point], layout=layout)
        else:
            fig = go.Figure(data=[trace, acceptance_area, rejection_area, critical_point, test_statistic_point], layout=layout)

        # Devolver los resultados
        if varianza_conocida == 'si':
            return render_template('index.html', z=z, resultado=resultado, graph_json=pio.to_json(fig), z_alpha=z_alpha, varianza_conocida=varianza_conocida)
        else:
            return render_template('index.html', t=t, resultado=resultado, graph_json=pio.to_json(fig), t_alpha=t_alpha, varianza_conocida=varianza_conocida)
        
    except Exception as e:
        print("Error:", e)
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
