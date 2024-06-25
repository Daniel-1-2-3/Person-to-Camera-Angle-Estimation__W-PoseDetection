import { useState, useEffect, useRef } from 'react'
import { StyleSheet, Text, View, StatusBar, useWindowDimensions, Image, Button } from 'react-native';
import { TabView, SceneMap, TabBar } from 'react-native-tab-view';

const FirstRoute = () => (
    <View style={{ flex: 1, backgroundColor: '#000359' }} />
);

const SecondRoute = () => {
    <View style={{ flex: 1, backgroundColor: '#673ab7' }} />
}

const renderScene = SceneMap({
    cam: FirstRoute,
    sim: SecondRoute,
  });

const rendertabBar = (props) => ( //props allow other functional components to pass inputs to this component
  <View>
    <View style={{height: 30, backgroundColor: 'black'}}/>
    <TabBar {...props} style={{backgroundColor: 'black'}}/>
  </View>

)
const App = () => {
    const layout = useWindowDimensions();
    const [index, setIndex] = useState(0);
    const [routes] = useState([
        { key: 'cam', title: 'Camera' },
        { key: 'sim', title: 'Simulation' }
    ]);

    return (
        <TabView
            navigationState={{ index, routes }}
            renderScene={renderScene}
            renderTabBar={rendertabBar}
            onIndexChange={setIndex}
            initialLayout={{ width: layout.width }}
        />
    );
};

const styles = StyleSheet.create({
    frame: {
        zIndex: 1,
        width: '100%',
        height: '90%',
        top: 0,
    },
});

export default App;