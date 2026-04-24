async function calcularRiesgo() {
  const data = {
    pregnancies: parseFloat(document.getElementById("pregnancies").value),
    glucose: parseFloat(document.getElementById("glucose").value),
    blood_pressure: parseFloat(document.getElementById("blood_pressure").value),
    skin_thickness: parseFloat(document.getElementById("skin_thickness").value),
    insulin: parseFloat(document.getElementById("insulin").value),
    bmi: parseFloat(document.getElementById("bmi").value),
    diabetes_pedigree_function: parseFloat(document.getElementById("dpf").value),
    age: parseFloat(document.getElementById("age").value)
  };

  // Mostrar estado de carga
  const btn = document.getElementById("btn-calcular");
  btn.querySelector(".btn-text").style.display = "none";
  btn.querySelector(".btn-spinner").style.display = "inline";
  btn.disabled = true;

  // Ocultar resultados anteriores
  document.getElementById("resultado").style.display = "none";
  document.getElementById("explicacion-contenedor").style.display = "none";

  try {
    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    });

    const result = await response.json();

    if (result.error) {
      mostrarError("Error del servidor: " + result.error);
      return;
    }

    const porcentaje = result.probability ? (result.probability * 100).toFixed(1) : 0;
    const riesgo = result.risk || "No disponible";

    // Determinar nivel de riesgo
    let nivel = "bajo";
    let icono = "✅";
    if (result.probability >= 0.5) {
      nivel = "alto";
      icono = "⚠️";
    } else if (result.probability >= 0.2) {
      nivel = "moderado";
      icono = "🔶";
    }

    // Mostrar resultado principal
    const resultadoDiv = document.getElementById("resultado");
    resultadoDiv.style.display = "block";
    resultadoDiv.className = `resultado riesgo-${nivel}`;

    document.getElementById("resultado-icono").textContent = icono;
    document.getElementById("resultado-porcentaje").textContent = `${porcentaje}% de probabilidad`;
    document.getElementById("resultado-riesgo").textContent = riesgo;

    // Barra de progreso
    const barra = document.getElementById("barra-progreso");
    barra.style.width = "0%";
    barra.className = `barra-progreso barra-${nivel}`;
    setTimeout(() => {
      barra.style.width = `${porcentaje}%`;
    }, 50);

    // Mostrar explicación por factor
    if (result.explicacion && result.explicacion.length > 0) {
      const lista = document.getElementById("explicacion-lista");
      lista.innerHTML = "";

      result.explicacion.forEach((factor, index) => {
        const tarjeta = document.createElement("div");
        tarjeta.className = `factor-card factor-${factor.estado}`;
        tarjeta.style.animationDelay = `${index * 0.08}s`;

        const estadoLabel = {
          alto: "⚠️ Alto",
          moderado: "🔶 Moderado",
          normal: "✅ Normal"
        }[factor.estado] || factor.estado;

        tarjeta.innerHTML = `
          <div class="factor-header">
            <span class="factor-nombre">${factor.variable}</span>
            <span class="factor-badge badge-${factor.estado}">${estadoLabel}</span>
          </div>
          <div class="factor-mensaje">${factor.mensaje}</div>
        `;
        lista.appendChild(tarjeta);
      });

      document.getElementById("explicacion-contenedor").style.display = "block";
    }

  } catch (error) {
    mostrarError("Error al conectar con el servidor. Verifica que la API esté corriendo.");
  } finally {
    btn.querySelector(".btn-text").style.display = "inline";
    btn.querySelector(".btn-spinner").style.display = "none";
    btn.disabled = false;
  }
}

function mostrarError(msg) {
  const resultadoDiv = document.getElementById("resultado");
  resultadoDiv.style.display = "block";
  resultadoDiv.className = "resultado riesgo-error";
  resultadoDiv.innerHTML = `<div style="text-align:center; padding: 10px;">❌ ${msg}</div>`;
}
