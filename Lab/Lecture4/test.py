import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

class DeadNeuronVisualizer(tf.keras.callbacks.Callback):
    """
    Callback per visualizzare l'evoluzione dei neuroni morti durante l'addestramento
    """
    def __init__(self, validation_data, save_plots=True, plot_freq=50):
        super().__init__()
        self.x_val, self.y_val = validation_data
        self.save_plots = save_plots
        self.plot_freq = plot_freq
        
        # Storici per i grafici
        self.dead_history = []
        self.epochs = []
        self.final_activations = None
        
    def on_train_begin(self, logs=None):
        print("\n🔍 Inizializzazione monitoraggio neuroni morti...")
        # Crea il modello di attivazione all'inizio del training
        self._create_activation_model()
        
    def _create_activation_model(self):
        """Crea il modello per estrarre le attivazioni"""
        # Trova i layer Dense con attivazione ReLU
        self.relu_layers = []
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                # Verifica se l'attivazione è ReLU
                if hasattr(layer, 'activation') and hasattr(layer.activation, '__name__'):
                    if layer.activation.__name__ == 'relu':
                        self.relu_layers.append(layer)
                # Alternativa: controlla il nome dell'attivazione
                elif hasattr(layer, 'activation') and str(layer.activation).find('relu') != -1:
                    self.relu_layers.append(layer)
        
        if not self.relu_layers:
            print("⚠️ Nessun layer con attivazione ReLU trovato!")
            return
            
        # Crea il modello di attivazione
        self.activation_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=[layer.output for layer in self.relu_layers]
        )
        
    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch)
        
        # Assicurati che il modello di attivazione sia stato creato
        if not hasattr(self, 'activation_model'):
            self._create_activation_model()
            
        try:
            # Analizza attivazioni
            activations = self.activation_model.predict(self.x_val, verbose=0)
            
            # Salva le attivazioni finali
            if epoch == self.params['epochs'] - 1:
                self.final_activations = activations
            
            # Conta neuroni morti per layer
            epoch_dead = []
            for layer_acts in activations:
                dead_in_layer = np.sum(np.sum(layer_acts > 0, axis=0) == 0)
                epoch_dead.append(dead_in_layer)
            
            self.dead_history.append(epoch_dead)
            
            # Visualizzazione periodica
            if epoch % self.plot_freq == 0 or epoch == self.params['epochs'] - 1:
                self._plot_current_state(epoch, activations, epoch_dead)
                
        except Exception as e:
            print(f"⚠️ Errore durante l'analisi delle attivazioni: {e}")
    
    def _plot_current_state(self, epoch, activations, dead_counts):
        """Crea visualizzazione dello stato attuale"""
        if len(activations) == 0:
            print("⚠️ Nessuna attivazione da visualizzare")
            return
            
        n_layers = len(activations)
        fig = plt.figure(figsize=(16, 4 * n_layers))
        gs = GridSpec(n_layers, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        for i, (layer_acts, dead_count) in enumerate(zip(activations, dead_counts)):
            n_neurons = layer_acts.shape[1]
            n_samples = layer_acts.shape[0]
            
            # 1. Heatmap delle attivazioni
            ax1 = fig.add_subplot(gs[i, 0])
            # Normalizza per migliorare visualizzazione
            if np.any(layer_acts > 0):
                vmax = np.percentile(layer_acts[layer_acts > 0], 95)
            else:
                vmax = 1
            im = ax1.imshow(layer_acts.T, aspect='auto', cmap='RdBu_r', 
                          vmin=-vmax*0.1, vmax=vmax)
            ax1.set_title(f'Layer {i+1} - Attivazioni\n(Morti: {dead_count}/{n_neurons})', 
                         fontsize=10)
            ax1.set_ylabel('Neuroni')
            ax1.set_xlabel('Campioni')
            plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
            
            # Evidenzia neuroni morti
            dead_neurons = np.sum(layer_acts > 0, axis=0) == 0
            if np.any(dead_neurons):
                dead_indices = np.where(dead_neurons)[0]
                for idx in dead_indices:
                    ax1.axhline(y=idx - 0.5, color='red', linewidth=2, linestyle='--')
                    ax1.axhline(y=idx + 0.5, color='red', linewidth=2, linestyle='--')
            
            # 2. Percentuale attivazione per neurone
            ax2 = fig.add_subplot(gs[i, 1])
            activation_rate = (np.sum(layer_acts > 0, axis=0) / n_samples) * 100
            colors = ['red' if rate == 0 else 'green' for rate in activation_rate]
            bars = ax2.bar(range(n_neurons), activation_rate, color=colors, alpha=0.7)
            ax2.set_ylim(0, 100)
            ax2.set_title(f'Layer {i+1} - % Attivazione per Neurone', fontsize=10)
            ax2.set_xlabel('Indice Neurone')
            ax2.set_ylabel('% Campioni con attivazione > 0')
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            
            # 3. Istogramma attivazioni
            ax3 = fig.add_subplot(gs[i, 2])
            active_vals = layer_acts[layer_acts > 0]
            if len(active_vals) > 0:
                ax3.hist(active_vals, bins=50, color='blue', alpha=0.7, edgecolor='black')
                ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Soglia ReLU')
                ax3.set_title(f'Layer {i+1} - Distribuzione attivazioni\n(Media={np.mean(active_vals):.3f})', 
                            fontsize=10)
                ax3.legend()
            else:
                ax3.text(0.5, 0.5, 'TUTTI I NEURONI MORTI!', 
                        transform=ax3.transAxes, ha='center', va='center', 
                        fontsize=12, color='red', fontweight='bold')
                ax3.set_title(f'Layer {i+1} - NESSUN NEURONE ATTIVO!', fontsize=10)
            ax3.set_xlabel('Valore attivazione')
            ax3.set_ylabel('Frequenza')
        
        plt.suptitle(f'Stato Neuroni - Epoca {epoch}', fontsize=14, fontweight='bold')
        
        if self.save_plots:
            plt.savefig(f'dead_neurons_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
            print(f"📊 Grafico salvato: dead_neurons_epoch_{epoch}.png")
            
    def on_train_end(self, logs=None):
        """Crea grafici finali dell'evoluzione dei neuroni morti"""
        if not self.dead_history or self.final_activations is None:
            print("⚠️ Nessun dato disponibile per il report finale")
            return
        
        dead_history = np.array(self.dead_history)
        n_layers = dead_history.shape[1]
        
        fig, axes = plt.subplots(1, n_layers, figsize=(5*n_layers, 5))
        if n_layers == 1:
            axes = [axes]
        
        for i in range(n_layers):
            ax = axes[i]
            ax.plot(self.epochs, dead_history[:, i], 'b-', linewidth=2, label='Neuroni morti')
            ax.fill_between(self.epochs, 0, dead_history[:, i], alpha=0.3)
            ax.set_xlabel('Epoca', fontsize=12)
            ax.set_ylabel('Numero neuroni morti', fontsize=12)
            ax.set_title(f'Layer {i+1}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Aggiungi linea per mostrare la soglia di morte totale
            if i < len(self.final_activations):
                total_neurons = self.final_activations[i].shape[1]
                ax.axhline(y=total_neurons, color='r', linestyle='--', alpha=0.5, 
                          label=f'Totale neuroni: {total_neurons}')
                ax.legend()
        
        plt.suptitle('Evoluzione Neuroni Morti Durante Training', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig('dead_neurons_evolution.png', dpi=150, bbox_inches='tight')
            print("\n📊 Grafico evoluzione salvato: dead_neurons_evolution.png")
        
        
        # Report finale
        print("\n" + "="*60)
        print("📊 REPORT FINALE NEURONI MORTI")
        print("="*60)
        for i, layer_acts in enumerate(self.final_activations):
            n_neurons = layer_acts.shape[1]
            dead_neurons = np.sum(np.sum(layer_acts > 0, axis=0) == 0)
            alive_neurons = n_neurons - dead_neurons
            print(f"\n🔷 Layer {i+1}:")
            print(f"   Neuroni totali: {n_neurons}")
            print(f"   ✅ Neuroni vivi: {alive_neurons}")
            print(f"   💀 Neuroni morti: {dead_neurons} ({dead_neurons/n_neurons*100:.1f}%)")
            
            if dead_neurons > 0:
                dead_indices = np.where(np.sum(layer_acts > 0, axis=0) == 0)[0]
                print(f"   📍 Indici morti: {dead_indices}")
            
            # Calcola statistiche attivazioni
            active_vals = layer_acts[layer_acts > 0]
            if len(active_vals) > 0:
                print(f"   📊 Attivazione media (quando attivi): {np.mean(active_vals):.4f}")
                print(f"   📊 Deviazione std attivazioni: {np.std(active_vals):.4f}")
        
        print("\n" + "="*60)


# ============================================================================
# ESEMPIO DI UTILIZZO CORRETTO
# ============================================================================

# Prepara i dati (usa i tuoi dati reali)
from sklearn.preprocessing import StandardScaler

# Normalizza i dati (importante per ReLU)
scaler_x = StandardScaler()
scaler_y = StandardScaler()


data = np.loadtxt('data.dat')

x_tr = data[:,0]
y_tr = data[:,1]
x_val = data[:,2]
y_val = data[:,3]

x_tr_norm = scaler_x.fit_transform(x_tr.reshape(-1, 1))
y_tr_norm = scaler_y.fit_transform(y_tr.reshape(-1, 1))
x_val_norm = scaler_x.transform(x_val.reshape(-1, 1))
y_val_norm = scaler_y.transform(y_val.reshape(-1, 1))

# Crea il modello con ReLU
model_relu = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(10, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.Dense(10, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.Dense(10, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.Dense(1, activation='linear')
])

model_relu.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), 
                   loss=tf.keras.losses.MSE)

# Crea il visualizzatore (NON chiamare fit prima di questo!)
dead_viz = DeadNeuronVisualizer(
    validation_data=(x_val_norm, y_val_norm),
    save_plots=True,
    plot_freq=50
)

# Ora allena il modello con il callback
history = model_relu.fit(
    x_tr_norm, y_tr_norm,
    validation_data=(x_val_norm, y_val_norm),
    epochs=500,
    batch_size=32,
    callbacks=[dead_viz],
    verbose=1
)